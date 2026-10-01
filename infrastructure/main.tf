terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Remote backend for state file (highly recommended for production)
  # backend "s3" {
  #   bucket = "advertest-terraform-state"
  #   key    = "prod/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region
}

# ==============================================================================
# S3 Data Lake (For Ego4D/WIDER FACE Images, HITL insights, Adversarial pools)
# ==============================================================================
resource "aws_s3_bucket" "data_lake" {
  bucket = var.s3_bucket_name
}

resource "aws_s3_bucket_versioning" "data_lake_versioning" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Apply lifecycle policy to transition old adversarial pools to Glacier
resource "aws_s3_bucket_lifecycle_configuration" "data_lake_lifecycle" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    id     = "archive-old-pools"
    status = "Enabled"
    
    filter {
      prefix = "data/pools/"
    }

    transition {
      days          = 30
      storage_class = "GLACIER"
    }
  }
}

# ==============================================================================
# VPC & Security Groups
# ==============================================================================
# Using Default VPC for this boilerplate. In strict prod, create a custom VPC.
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "advertest_sg" {
  name        = "advertest_production_sg"
  description = "Security group for AdverTest End-to-End infrastructure"
  vpc_id      = data.aws_vpc.default.id

  # Allow external HTTP traffic for Frontend
  ingress {
    description = "HTTP Frontend"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow external HTTPS traffic for Frontend
  ingress {
    description = "HTTPS Frontend"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow internal routing to FastAPI backend
  ingress {
    description = "FastAPI Backend (Internal)"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    self        = true # Only allow traffic from instances in the same SG
  }

  # Allow SSH strictly from whitelisted IP
  ingress {
    description = "SSH Access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_ip]
  }

  # Allow all outbound traffic (to pull Docker images, contact VLM APIs, etc)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "AdverTest-SG"
  }
}

# ==============================================================================
# Compute: EC2 Deep Learning Instance for GPU workload (YOLO, 3DGS, Ray)
# ==============================================================================
# Lookup the latest AWS Deep Learning AMI (Ubuntu 22.04)
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }
}

resource "aws_instance" "advertest_backend_gpu" {
  ami           = data.aws_ami.deep_learning.id
  instance_type = var.gpu_instance_type # Default: g4dn.xlarge (NVIDIA T4)

  vpc_security_group_ids = [aws_security_group.advertest_sg.id]
  key_name               = var.ssh_key_name

  # Provision sufficient root storage for Docker images and Datasets
  root_block_device {
    volume_size = 250 
    volume_type = "gp3"
  }

  tags = {
    Name = "AdverTest-Production-GPU"
  }

  # User Data script to pull and run the Docker Compose stack on boot
  user_data = <<-EOF
              #!/bin/bash
              echo "Initializing AdverTest Production Server..."
              
              # Pull the latest docker-compose.yml from S3 or Git
              # Start the stack
              # docker-compose up -d
              EOF
}
