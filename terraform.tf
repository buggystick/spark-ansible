terraform {
  required_version = ">= 1.6"

  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = ">= 1.47"
    }
  }
}

variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

module "talos" {
  source  = "hcloud-talos/talos/hcloud"
  version = "v2.21.0"

  talos_version            = "v1.11.0"
  hcloud_token             = var.hcloud_token
  disable_arm              = true
  firewall_use_current_ip  = true
  cluster_name             = "dsreed"
  datacenter_name          = "hil-dc1"
  control_plane_count      = 3
  control_plane_server_type = "cpx11"
  # omit worker_count / worker_nodes to create control planes only
}
