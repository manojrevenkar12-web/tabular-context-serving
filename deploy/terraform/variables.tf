variable "project_id" {
  description = "GCP project that hosts the serving plane."
  type        = string
}

variable "region" {
  description = "Region for the cluster, the dataset bucket and the key ring."
  type        = string
  default     = "europe-west4"
}

variable "cluster_name" {
  description = "Name of the GKE cluster."
  type        = string
  default     = "tcs"
}

variable "namespace" {
  description = "Kubernetes namespace the Helm chart is installed into."
  type        = string
  default     = "tcs"
}

variable "dataset_retention_days" {
  description = "Uploaded datasets older than this are deleted by the bucket itself, whatever the API does."
  type        = number
  default     = 30
}

variable "gpu_pools" {
  description = "One autoscaled node pool per GPU kind the plane schedules on (config/platform.toml [[fleet]])."
  type = map(object({
    kind              = string
    machine_type      = string
    accelerator       = string
    accelerator_count = number
    zones             = list(string)
    min_nodes         = number
    max_nodes         = number
    spot              = bool
  }))
  default = {
    l4 = {
      kind              = "l4"
      machine_type      = "g2-standard-8"
      accelerator       = "nvidia-l4"
      accelerator_count = 1
      zones             = ["europe-west4-a", "europe-west4-b"]
      min_nodes         = 0
      max_nodes         = 4
      spot              = false
    }
    a100 = {
      kind              = "a100"
      machine_type      = "a2-ultragpu-1g"
      accelerator       = "nvidia-a100-80gb"
      accelerator_count = 1
      zones             = ["europe-west4-a"]
      min_nodes         = 0
      max_nodes         = 2
      spot              = false
    }
    h100 = {
      kind              = "h100"
      machine_type      = "a3-highgpu-1g"
      accelerator       = "nvidia-h100-80gb"
      accelerator_count = 1
      zones             = ["europe-west4-b"]
      min_nodes         = 0
      max_nodes         = 2
      spot              = false
    }
  }
}
