# Use the official Modular MAX NVIDIA container
FROM modular/max-nvidia-full:latest

# Set working directory
WORKDIR /app

# Create cache directory and fix permissions
RUN mkdir -p /app/.cache && chmod -R 777 /app

# Point Hugging Face cache to the writable folder
ENV HF_HOME=/app/.cache

# Expose the standard Hugging Face port
EXPOSE 7860

# Reset entrypoint to avoid conflicts
ENTRYPOINT []

# COMMAND:
# 1. Removed '--host' (it defaults to 0.0.0.0 automatically)
# 2. Kept '--port 7860' (required for Spaces)
CMD ["max", "serve", "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct", "--port", "7860"]