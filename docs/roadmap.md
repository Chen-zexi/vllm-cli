# vLLM CLI Roadmap

## Serving Configurations
- [x] **Environment Variable Support** - Comprehensive three-tier environment variable system for complete control
- [x] **GPU Selection Feature** - Select specific GPUs for model serving via `--device` flag or interactive UI
- [x] **Server Cleanup Control** - Configure whether servers are stopped when CLI exits
- [x] **vLLM native Arguments** - Added 16+ new critical vLLM arguments for advanced configurations
- [x] **Shortcuts System** - Save and quickly launch model+profile combinations
- [ ] **Docker Backend Support** - Use existing vLLM Docker images as backend

## UI Features
- [x] **Rich Terminal UI** - Rich terminal interface with menu-driven navigation
- [x] **Command-Line Mode** - Direct CLI commands for automation and scripting
- [x] **System Information** - GPU, memory, and CUDA compatibility checking
- [ ] **CPU Stats** - CPU usage, memory usage, and disk usage checking
- [ ] **UI Customization**
    - [x] Customize GPU stats bar
    - [x] Customize log refresh frequency
    - [ ] Customize theme
    - [ ] Multi-language support
- [ ] **Server Monitoring**
    - [x] Real-time monitoring of active vLLM servers
    - [ ] Server monitoring after exiting program
- [ ] **Log Viewer**
    - [x] View the complete log file when server startup fails
    - [ ] View logs for past runs

## Hardware Support
- [x] **NVIDIA GPUs** - Support for NVIDIA GPUs
- [ ] **AMD GPUs** - Support for AMD GPUs (ROCm) --> Need help from AMD users! Contributions are welcome!

## Model Discovery Support
- [x] **HuggingFace Model Support** - Discover and serve models from HuggingFace Hub
- [x] **Custom Model Directories** - Support for custom model directories
- [x] **Ollama Model Support** - Discover and serve GGUF models from Ollama directories (experimental)
- [x] **GGUF file loading** - Support for direct GGUF file loading
- [x] **Model Manifest Support** - Map custom models in vLLM CLI native way with `models_manifest.json`
- [ ] **Oracle Cloud Infrastructure (OCI) Registry** - Support for OCI Registry format

## Future Enhancements

Additional features and improvements planned for future releases will be added here as the project evolves.
