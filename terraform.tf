terraform {
  required_version = ">= 1.6"

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = ">= 1.47"
    }
  }
}

module "talos" {
  source  = "hcloud-talos/talos/hcloud"
  version = "v2.21.0"

  talos_version             = "v1.11.0"
  hcloud_token              = var.hcloud_token
  disable_arm               = true
  firewall_use_current_ip   = true
  firewall_kube_api_source  = ["0.0.0.0/0"]
  cluster_name              = var.cluster_name
  cluster_api_host          = var.cluster_api_host
  output_mode_config_cluster_endpoint = "cluster_endpoint"
  enable_alias_ip           = false
  datacenter_name           = var.datacenter_name
  control_plane_count       = 3
  control_plane_server_type = "cpx11"
  # omit worker_count / worker_nodes to create control planes only
}

provider "hcloud" {
  token = var.hcloud_token
}

variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

variable "cluster_name" {
  description = "Cluster name used for labels and resources"
  type        = string
  default     = "dsreed"
}

variable "cluster_api_host" {
  description = "DNS name for Kubernetes API (points to LB IP)"
  type        = string
  default     = "k8s.dsreed.net"
}

variable "datacenter_name" {
  description = "Hetzner datacenter (e.g., hil-dc1)"
  type        = string
  default     = "hil-dc1"
}

output "talosconfig" {
  value     = module.talos.talosconfig
  sensitive = true
}

output "kubeconfig" {
  value     = local.kubeconfig_lb
  sensitive = true
}

# Load balancer for Kubernetes API
locals {
  lb_location = substr(var.datacenter_name, 0, 3) # e.g., "hil" from "hil-dc1"
}

resource "hcloud_load_balancer" "k8s_api" {
  name                = "${var.cluster_name}-k8s-api"
  load_balancer_type  = "lb11"
  location            = local.lb_location
  delete_protection   = false
}

resource "hcloud_load_balancer_network" "k8s_api" {
  load_balancer_id = hcloud_load_balancer.k8s_api.id
  network_id       = module.talos.hetzner_network_id
}

resource "hcloud_load_balancer_service" "k8s_api" {
  load_balancer_id = hcloud_load_balancer.k8s_api.id
  protocol         = "tcp"
  listen_port      = 6443
  destination_port = 6443

  health_check {
    protocol = "tcp"
    port     = 6443
    interval = 15
    timeout  = 10
    retries  = 3
  }
}

resource "hcloud_load_balancer_target" "control_planes" {
  load_balancer_id = hcloud_load_balancer.k8s_api.id
  type             = "label_selector"
  label_selector   = "cluster=${var.cluster_name},role=control-plane"
  use_private_ip   = true
  depends_on       = [hcloud_load_balancer_network.k8s_api]
}

output "k8s_api_lb_ipv4" {
  value       = hcloud_load_balancer.k8s_api.ipv4
  description = "Public IPv4 of the Kubernetes API load balancer"
}

output "k8s_api_endpoint" {
  value       = "${hcloud_load_balancer.k8s_api.ipv4}:6443"
  description = "Kubernetes API endpoint via load balancer"
}

locals {
  kubeconfig_lb = replace(
    module.talos.kubeconfig,
    module.talos.kubeconfig_data.host,
    "https://${var.cluster_api_host}:6443"
  )
}
