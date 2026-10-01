output "data_lake_bucket_name" {
  description = "The name of the provisioned S3 Data Lake bucket"
  value       = aws_s3_bucket.data_lake.id
}

output "backend_gpu_public_ip" {
  description = "The public IP address of the EC2 GPU Instance"
  value       = aws_instance.advertest_backend_gpu.public_ip
}

output "ssh_connection_string" {
  description = "Command to SSH into the production server"
  value       = "ssh -i ~/.ssh/${var.ssh_key_name}.pem ubuntu@${aws_instance.advertest_backend_gpu.public_ip}"
}
