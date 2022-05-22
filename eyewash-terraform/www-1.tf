resource "digitalocean_droplet" "web" {
  image = "ubuntu-18-04-x64"
  name = "www-1"
  region = "nyc3"
  size = "s-4vcpu-8gb"
  private_networking = true
  ssh_keys = [
    data.digitalocean_ssh_key.do_key.id
  ]
  connection {
    host = self.ipv4_address
    user = "root"
    type = "ssh"
    private_key = file(var.pvt_key)
    timeout = "2m"
  }
  provisioner "remote-exec" {
    inline = [
      "export PATH=$PATH:/usr/bin",
      # install nginx
      "sudo apt-get update",
      # "sudo apt-get -y install nginx"
    ]
  }
}