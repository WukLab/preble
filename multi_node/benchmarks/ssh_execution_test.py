import paramiko
import re, time
import requests

class SSHRuntime:
    def __init__(self, model_path, ssh_config, gpu, **kwargs):
        self.model_path = model_path
        self.ssh_config = ssh_config
        self.gpu = gpu
        assert self.ssh_config
        self.ssh_client = self.initialize_ssh_client()
        self.port = self.start_remote_runtime()
        print("Started remote server 1 on port")
        # Initialize server with running these configs
        # Save url to url
        super().__init__()

    def initialize_ssh_client(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**self.ssh_config)
        return ssh

    def start_remote_runtime(self, **kwargs):
        cli_args = self.kwargs_to_cli_args(**kwargs)
        environment_variables = {'CUDA_VISIBLE_DEVICES': str(self.gpu)}
        command = f'setsid /mnt/ssd1/vikranth/sglang_experiments/sglang_env/bin/python -m sglang.launch_server --model-path {self.model_path} {cli_args} --host 0.0.0.0'
        print("Running command", command)
        transport = self.ssh_client.get_transport()
        channel = transport.open_session()
        self.channel = channel
        channel.update_environment(environment_variables)
        channel.exec_command(command)
        stdout = channel.makefile('r', -1)
        stderr = channel.makefile_stderr('r', -1)
        timeout = 40
        end_time = time.time() + timeout
        port = None
        while time.time() < end_time:
            line = stdout.readline()  # Read line from stdout
            if not line:
                line = stderr.readline()  # If stdout is empty, try to read from stderr
            print(line, end='')  # Optional: print the line for debugging
            
            # Search for the port number in the line
            match = re.search(r"Server is on port (\d+) on host (.*) on pid (\d+)", line)
            if match:
                port = match.group(1)  # Capture the port number
                pid = match.group(3)
                break

            if not line:
                time.sleep(1)  # Wait before trying to read more output

        if not port:
            raise Exception("Failed to detect server startup within the timeout period.")
        self.url = f"http://{self.ssh_config['hostname']}:{port}"

        # Wait for the /model_info to return valid json for 5 attempts
        for _ in range(5):
            try:
                response = requests.get(f"{self.url}/get_model_info")
                if response.status_code == 200:
                    break
                time.sleep(2)
            except Exception as e:
                print(e)
            time.sleep(1)
        print(response.json())
        self.process_pid = pid
        self.port = port

        return port   
# Wait for the output to be available and read the PID
    
    def shutdown(self):
        if self.ssh_client:
            self.ssh_client.exec_command(f"pkill -f {self.port}")
            self.ssh_client.exec_command(f"pkill -f sglang")
            self.ssh_client.close()

    def kwargs_to_cli_args(self, **kwargs):
        args = []
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key.replace('_', '-')}")
            else:
                args.append(f"--{key.replace('_', '-')} {value}")
        return ' '.join(args)

if __name__ == "__main__":
    ssh_config = {
        "hostname": "192.168.1.18",
        "username": "vikranth",
        "port": 456,
    }
    runtime = SSHRuntime("mistralai/Mistral-7B-v0.1", ssh_config, 0, cuda_devices=0)
    print(f"Running on port {runtime.port}")
    time.sleep(30)
    runtime.shutdown()
    