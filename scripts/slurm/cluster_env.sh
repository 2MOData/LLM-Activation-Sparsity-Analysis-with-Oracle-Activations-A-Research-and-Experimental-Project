# Environment for the school cluster, sourced by the Slurm jobs and by ~/.bashrc on the login node.
#
# Slurm jobs cannot read the home folder (/homes is a Kerberos-protected network disk and a job
# carries no ticket), so everything a job needs lives on /Brain: uv itself, the Python it
# installs (which .venv points to), and the caches. The login node must use the same paths,
# otherwise `uv sync` builds a .venv that points into the home folder.

PROCOM_DIR=/Brain/private/ProCom-AO

export PATH="$PROCOM_DIR/tools/bin:$PATH"                # uv
export UV_PYTHON_INSTALL_DIR="$PROCOM_DIR/tools/python"   # the Python that .venv points to
export UV_CACHE_DIR="$PROCOM_DIR/cache/uv"
export HF_HOME="$PROCOM_DIR/cache/huggingface"            # model weights
