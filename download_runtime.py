import subprocess

# Download ONNX models
subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe', '-O', 'AnimeGAN_Hayao.onnx'])
subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1', '-O', 'AnimeGANv2_Hayao.onnx'])
subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK', '-O', 'AnimeGANv2_Shinkai.onnx'])
subprocess.run(['wget', '--no-check-certificate', 'https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI', '-O', 'AnimeGANv2_Paprika.onnx'])
