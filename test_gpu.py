import torch
import sys

def test_gpu_available():
    """
    Checks if a GPU is available and performs a simple calculation.
    """
    print("Verifica che PyTorch veda la GPU...")

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Errore: Nessuna GPU rilevata da PyTorch. L'accelerazione GPU non è disponibile.")
        sys.exit(1)  # Exit with a non-zero code to indicate failure

    print("PyTorch ha rilevato una GPU!")
    print(f"Nome della GPU: {torch.cuda.get_device_name(0)}")
    print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")

    try:
        # Perform a simple calculation on the GPU
        print("Eseguo un calcolo di prova sulla GPU...")
        a = torch.randn(2, 3, device='cuda')
        b = torch.randn(2, 3, device='cuda')
        result = a @ b.T
        print("Risultato del calcolo di prova:")
        print(result)
        print("\nTest GPU completato con successo!")
        sys.exit(0) # Exit with a zero code to indicate success
    except Exception as e:
        print(f"Errore durante l'esecuzione del calcolo sulla GPU: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gpu_available()