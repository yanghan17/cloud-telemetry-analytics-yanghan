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
