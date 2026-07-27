variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "ap-southeast-1" # Singapore - closest region, matches Cloud Engine Digital HQ
}

variable "project_name" {
  description = "Project name used as a prefix for resource naming"
  type        = string
  default     = "cloud-telemetry"
}

variable "owner" {
  description = "Owner tag for cost tracking and resource identification"
  type        = string
  default     = "yanghan"
}

variable "db_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "telemetry"
}

variable "db_username" {
  description = "Master username for the RDS instance"
  type        = string
  default     = "telemetry_admin"
}

variable "allowed_ip_cidr" {
  description = "Your local public IP (as a /32 CIDR) allowed to connect to RDS. Get it via: curl checkip.amazonaws.com"
  type        = string
  # No default on purpose -- forces you to explicitly set this rather than
  # accidentally leaving a wide-open default in version control.
}
