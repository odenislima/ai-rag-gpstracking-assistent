# free_gpu_memory.sh

#!/bin/bash

# Script para limpar a memoria da GPU matando processos Python pesados
# e reativar automaticamente o ambiente Conda desejado

# Parametros
ENV_NAME="rag-faiss"
MEMORY_THRESHOLD_MB=500  # Limite de memoria em MiB para considerar processo pesado

echo "[INFO] Listando processos usando a GPU..."
nvidia-smi

echo "[INFO] Procurando processos Python que consomem mais de ${MEMORY_THRESHOLD_MB}MiB de GPU..."

# Pega PIDs de processos Python que ultrapassam o limite de memoria definido
PIDS=$(nvidia-smi | grep python | awk '{ if ($9 > '$MEMORY_THRESHOLD_MB') print $5 }')

if [ -z "$PIDS" ]; then
  echo "[INFO] Nenhum processo Python pesado encontrado na GPU. Nada para matar."
else
  echo "[INFO] Matando os seguintes PIDs: $PIDS"
  for PID in $PIDS; do
    kill -9 $PID
  done
  echo "[INFO] Processos terminados."
fi

# Reexibir o estado da GPU
sleep 2
nvidia-smi

# Reativar o ambiente Conda automaticamente
echo "[INFO] Ativando ambiente Conda: $ENV_NAME"

# Detecta Conda base
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate $ENV_NAME
else
    echo "[ERRO] Nao foi encontrado o script de inicializacao do Conda."
    echo "[ERRO] Ative o ambiente manualmente com: conda activate $ENV_NAME"
fi
