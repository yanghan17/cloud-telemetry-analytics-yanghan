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
