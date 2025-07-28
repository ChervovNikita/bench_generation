### Dataset generation

1. Clone the repo
```bash
git clone https://github.com/ChervovNikita/llm_detection_bench
```

2. Install requirements
```bash
cd llm_detection_bench
pip install -r requirements.txt
```

3. Install pm2 and jq packages
```bash
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
```

4. Install lshw, so ollama can detect GPU
```bash
apt install lshw -y
```

5. Install ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

6. Run ollama in background
```bash
pm2 start --name ollama "ollama serve"
```

7. Run the script
```bash
python -m src.main
```
