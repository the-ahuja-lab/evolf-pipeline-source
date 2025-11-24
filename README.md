# EvOlf: Source Code & Development Repository

Welcome to the source code repository for **EvOlf**, a deep-learning framework for predicting ligand-GPCR interactions.

> **Note for Users:** If you simply want to *run* the pipeline on your data, please visit the main pipeline repository:  
>    **[the-ahuja-lab/evolf-pipeline](https://github.com/the-ahuja-lab/evolf-pipeline)**
>
> This repository (`evolf-pipeline-source`) contains the raw scripts, Dockerfiles, environment configurations, and module definitions used to *build* and *maintain* the pipeline. It is intended for developers, contributors, and advanced users who wish to modify the underlying logic.

## Repository Structure

This repository is organized to separate the Nextflow orchestration logic from the scientific computation scripts and their environments.

```bash
.
├── main.nf                 # The main Nextflow orchestration script
├── nextflow.config         # Global configuration (Profiles, Docker, GPU settings)
│
├── modules/                # Nextflow Process Definitions (The "Wrappers")
│   └── local/
│       ├── prepare_input/  # Input standardization process
│       ├── chemberta/      # Wrapper for ChemBERTa featurization
│       ├── protbert/       # Wrapper for ProtBERT featurization
│       └── ... (wrappers for all 12+ processes)
│
└── envs/                   # The Core Logic & Environments (The "Guts")
    ├── EvOlf_DL/       # Shared Deep Learning Environment
    │   ├── Dockerfile      # Builds the 'evolfdl_env' image
    │   ├── env_gpu.lock.yml# Conda lock file for reproducibility
    │   ├── ChemBERTa.py    # Python script for ligand embedding
    │   ├── ProtBERT.py     # Python script for receptor embedding
    │   └── ProtT5.py       # Python script for receptor embedding
    │
    ├── signaturizer/       # Bioactivity Signature Environment
    │   ├── Dockerfile
    │   ├── Signaturizer.py
    │   └── ...
    │
    └── ... (folders for EvOlf_R, mol2vec, graph2vec, etc.)
```

## Development & Customization

### 1\. Modifying Scientific Logic (Python/R)

The actual scientific computation happens in the scripts located in `envs/`.

  * To change how ChemBERTa embeddings are generated, edit `envs/EvOlf_DL/ChemBERTa.py`.
  * To modify the final prediction model architecture, edit `envs/evolf_prediction/EvOlfPrediction.py`.

### 2\. Updating Environments (Docker/Conda)

We use a "sealed bubble" approach for reproducibility.

  * **Environments** are defined in `envs/<module>/<name>.yml`.
  * **Lock files** (`.yml`) are used in Dockerfiles to guarantee exact package versions.
  * **Dockerfiles** are located in each subdirectory of `envs/`.

**To update an environment:**

1.  Create recipe YAML file (e.g., `envs/signaturizer/Signaturizer.yml`).
2.  Re-generate the lock file using Conda.
3.  Re-build the Docker image:
    ```bash
    docker build -f envs/signaturizer/Dockerfile -t ahujalab/signaturizer_env:latest .
    ```
4.  Push the new image to the registry (required for the main pipeline to see changes).

### 3\. Modifying the Pipeline Flow (Nextflow)

The orchestration logic resides in `modules/` and `main.nf`.

  * **`modules/local/<process>/main.nf`**: Defines the input/output channels and the command string for a single step.
  * **`main.nf`**: Connects these modules into the final workflow (e.g., parallel fan-out, synchronization, fan-in).

## Container Images

This pipeline relies on pre-built images hosted on Docker Hub. The Dockerfiles in this repository correspond to these images:

| Repository | Description | Source Path |
| :--- | :--- | :--- |
| `ahujalab/evolfdl_env` | Shared DL environment (PyTorch, Transformers) | `envs/EvOlf_DL/` |
| `ahujalab/evolfprediction_env` | Prediction model environment | `envs/evolf_prediction/` |
| `ahujalab/evolfr_env` | R environment for data prep & compilation | `envs/EvOlf_R/` |
| `ahujalab/signaturizer_env` | TensorFlow env for Signaturizer | `envs/signaturizer/` |
| ... | (See `envs/` for full list) | ... |

## Testing Changes

To test your changes locally before pushing or updating the main pipeline:

1.  **Clone this repository:**

    ```bash
    git clone https://github.com/the-ahuja-lab/evolf-pipeline-source.git
    cd evolf-pipeline-source
    ```

2.  **Build Modified Images (Optional):**
    If you changed a script in `envs/`, build the local Docker image.
    *Note: You may need to update `nextflow.config` to point to your local image name or use `-with-docker <your-image>`.*

3.  **Run the Pipeline Locally:**

    ```bash
    nextflow run main.nf \
        --inputFile "test_data/sample.csv" \
        -profile docker,gpu
    ```

## Contributing

We welcome contributions\! If you wish to improve the model or add new featurizers:

1.  Fork this repository.
2.  Create a branch for your feature (`git checkout -b feature/AmazingFeaturizer`).
3.  Add your script and environment in a new folder under `envs/`.
4.  Create a Nextflow wrapper in `modules/local/`.
5.  Submit a Pull Request.

## License

The source code is available under the **MIT License**. See the `LICENSE` file for more details.

## Contact & Credits

The EvOlf Pipeline was developed by the **Ahuja Lab** at IIIT-Delhi.

  * **Principal Investigator:** [Dr. Gaurav Ahuja](https://github.com/the-ahuja-lab)
  * **Lead Developers:**
      * [Adnan Raza](https://github.com/woosflex)
      * [Syed Yasser](https://github.com/yasservision24)
      * [Pranjal Sharma](https://github.com/PRANJAL2208)
      * [Saveena Solanki](https://github.com/SaveenaSolanki)
      * [Ayushi Mittal](https://github.com/Aayushi006)