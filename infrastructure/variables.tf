variable "aws_region" {
  description = "The AWS region to deploy infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  description = "Name of the S3 bucket for AdverTest Data Lake"
  type        = string
  default     = "advertest-production-data-lake-v1"
}

variable "gpu_instance_type" {
  description = "EC2 Instance type for the ML backend"
  type        = string
  default     = "g4dn.xlarge" # Most cost-effective NVIDIA T4 instance
}

variable "admin_ip" {
  description = "Your local IP address to allow SSH access (Format: x.x.x.x/32)"
  type        = string
  # IMPORTANT: Change this to your actual IP via Terraform tfvars or CLI
  default     = "0.0.0.0/0" 
}

variable "ssh_key_name" {
  description = "Name of the AWS Key Pair to use for EC2 SSH access"
  type        = string
  default     = "advertest-prod-key"
}
