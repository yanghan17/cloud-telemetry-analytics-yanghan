terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
  # Local state for this solo project. For a team setup, this would point
  # to an S3 backend with DynamoDB state locking instead.
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project = var.project_name
      Owner   = var.owner
      Managed = "terraform"
    }
  }
}

# S3 bucket names must be globally unique across ALL AWS accounts.
# Rather than guessing a name and hoping it's free, generate a short
# random suffix so this always succeeds without manual trial-and-error.
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "telemetry_raw" {
  bucket = "${var.project_name}-${var.owner}-${random_id.bucket_suffix.hex}"
}

# Versioning: protects against a bad ingestion run overwriting good data --
# previous object versions remain recoverable.
resource "aws_s3_bucket_versioning" "telemetry_raw" {
  bucket = aws_s3_bucket.telemetry_raw.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Explicitly block all public access. S3 buckets are private by default,
# but this makes the intent explicit and prevents any future misconfigured
# bucket policy from accidentally exposing data.
resource "aws_s3_bucket_public_access_block" "telemetry_raw" {
  bucket = aws_s3_bucket.telemetry_raw.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption at rest using AWS-managed keys (SSE-S3).
# Satisfies the assessment's basic security requirements without the
# added complexity of managing customer KMS keys.
resource "aws_s3_bucket_server_side_encryption_configuration" "telemetry_raw" {
  bucket = aws_s3_bucket.telemetry_raw.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Reference the account's default VPC rather than building a custom one.
# S3 itself is not VPC-bound, but RDS (added in the next Terraform step)
# will need a VPC/subnet to launch into -- reusing the default VPC keeps
# this project's networking simple and explainable.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ------------------------------------------------------------------
# RDS PostgreSQL
# ------------------------------------------------------------------

# A dedicated subnet group tells RDS which subnets (within the default
# VPC) it's allowed to place the database into. RDS requires this even
# for a single-AZ instance -- it's how Multi-AZ failover would work if
# you ever enabled it later.
resource "aws_db_subnet_group" "telemetry" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = data.aws_subnets.default.ids
}

# Security group = firewall rules for the RDS instance. Only allow
# inbound Postgres (5432) from your own IP, not the whole internet.
resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg"
  description = "Allow Postgres access from a single trusted IP only"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "Postgres from my local machine"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Auto-generate a strong password. Never typed, hardcoded, or logged --
# only ever stored encrypted in Secrets Manager below.
resource "random_password" "db_password" {
  length  = 20
  special = false # avoids characters that sometimes break connection strings
}

resource "aws_db_instance" "telemetry" {
  identifier     = "${var.project_name}-postgres"
  engine         = "postgres"
  engine_version = "16"
  instance_class = "db.t4g.micro" # RDS Free Tier eligible

  allocated_storage = 20 # GB, Free Tier eligible
  storage_type       = "gp3"
  storage_encrypted  = true

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db_password.result

  db_subnet_group_name  = aws_db_subnet_group.telemetry.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = true # restricted to your IP via the security group above

  multi_az            = false # single-AZ: sufficient for a dev/assessment workload
  skip_final_snapshot = true  # disposable project infra, no production backup needed
  deletion_protection = false # allows `terraform destroy` to clean up when you're done
}

# ------------------------------------------------------------------
# Secrets Manager
# ------------------------------------------------------------------

# Store DB connection details as one JSON secret. Application code
# (ingestion, Streamlit dashboard, ML scripts) fetches this at runtime
# via boto3 instead of ever reading a password from a config file.
resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.project_name}/db-credentials"
  description = "PostgreSQL connection credentials for the telemetry project"
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.db_username
    password = random_password.db_password.result
    host     = aws_db_instance.telemetry.address
    port     = 5432
    dbname   = var.db_name
  })
}
