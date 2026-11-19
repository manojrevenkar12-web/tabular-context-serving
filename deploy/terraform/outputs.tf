output "cluster" {
  value = google_container_cluster.this.name
}

output "dataset_bucket" {
  value = google_storage_bucket.datasets.name
}

output "api_service_account" {
  description = "Annotate the chart's ServiceAccount with this (values.yaml serviceAccount.gcpServiceAccount)."
  value       = google_service_account.api.email
}

output "image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
