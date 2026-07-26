output "s3_bucket_name" {
  description = "Name of the S3 bucket for raw telemetry storage"
  value       = aws_s3_bucket.telemetry_raw.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.telemetry_raw.arn
}

output "default_vpc_id" {
  description = "ID of the default VPC being used"
  value       = data.aws_vpc.default.id
}

output "default_subnet_ids" {
  description = "IDs of the default VPC's subnets"
  value       = data.aws_subnets.default.ids
}
