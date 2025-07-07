import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob, os, random 

#---------- helpers ---------- 

def cargar_datos(path_healthy, path_parkinson, max_imgs=300, tam=(64, 64)):
    healthy = glob.glob(os.path.join(path_healthy, "*.png"))[:max_imgs] 
    park = glob.glob(os.path.join(path_parkinson, "*.png"))[:max_imgs] 
    X, y = [], [] 
    for f in healthy: 
        img = Image.open(f).convert("L").resize(tam) 
        X.append(np.asarray(img, dtype=np.float32).flatten())  # sin dividir por 255 
        y.append(0) 
    for f in park: 
        img = Image.open(f).convert("L").resize(tam) 
        X.append(np.asarray(img, dtype=np.float32).flatten()) 
        y.append(1) 
    
    print(f"Total imágenes: {len(X)}") 
    return np.array(X), np.array(y) 
  

def dividir_datos(X, y, test_ratio=0.2):
     idx = np.arange(len(X))
     np.random.shuffle(idx)
     n_test = int(len(idx) * test_ratio)
     return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]] 

def normalizar_datos(X_train, X_test):
     min_vals = np.min(X_train, axis=0)
     max_vals = np.max(X_train, axis=0)
     range_vals = max_vals - min_vals
     range_vals[range_vals == 0] = 1 
     X_train_norm = (X_train - min_vals) / range_vals
     X_test_norm = (X_test - min_vals) / range_vals
     return X_train_norm, X_test_norm 

#---------- función f ---------- 

def f_wb(x, w, b):
     z = np.dot(w, x) + b
     return (np.tanh(z) + 1) / 2 

#---------- pérdida y exactitud ---------- 

def loss(w, b, X, D):
     preds = np.array([f_wb(xi, w, b) for xi in X])
     return np.mean((preds - D) ** 2) 

def accuracy(w, b, X, D):
     preds = np.array([f_wb(xi, w, b) for xi in X])
     preds_bin = (preds >= 0.5).astype(int)
     return np.mean(preds_bin == D) 

#---------- gradientes ---------- 

def grad_L(w, b, X, D):
    grad_w = np.zeros_like(w)
    grad_b = 0.0
    for x_i, d_i in zip(X, D):
         z = np.dot(w, x_i) + b 
         tanh_z = np.tanh(z) 
         f = 0.5 * (1 + tanh_z) 
         common_term = (1 - tanh_z**2) * (f - d_i) 
         grad_w += common_term * x_i 
         grad_b += common_term 
    return grad_w, grad_b 

#---------- descenso por gradiente ---------- 

def gradient_descent(X_train, D_train, X_test, D_test, alpha, iterations=1000):
    K = X_train.shape[1] 
    w = np.random.randn(K) * 0.01 
    b = 0.0 

    history = { 
    'train_loss': [], 
    'train_acc': [], 
    'test_loss': [], 
    'test_acc': [] 
} 
 
    for it in range(iterations): 
        grad_w, grad_b = grad_L(w, b, X_train, D_train) 
        w -= alpha * grad_w 
        b -= alpha * grad_b 
    
        if it % 10 == 0 or it == iterations - 1: 
            train_loss = loss(w, b, X_train, D_train) 
            train_acc = accuracy(w, b, X_train, D_train) 
            test_loss = loss(w, b, X_test, D_test) 
            test_acc = accuracy(w, b, X_test, D_test) 
    
            history['train_loss'].append(train_loss) 
            history['train_acc'].append(train_acc) 
            history['test_loss'].append(test_loss) 
            history['test_acc'].append(test_acc) 
    
            print(f"Iter {it}: Train Loss={train_loss:.6f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.6f}, Test Acc={test_acc:.4f}") 
    
    return w, b, history 
  

#---------- main ---------- 

def main():
    path_h = "../Healthy" 
    path_p = "../Parkinson" 
    X, y = cargar_datos(path_h, path_p, max_imgs=300, tam=(64, 64)) 
    X_tr, X_te, y_tr, y_te = dividir_datos(X, y) 
    
    print("\n🔴 Entrenando con datos SIN normalizar...") 
    w, b, hist = gradient_descent(X_tr, y_tr, X_te, y_te, alpha=1e-6, iterations=1000) 
    
    print("\n🔵 Entrenando con datos NORMALIZADOS (min-max)...") 
    X_tr_norm, X_te_norm = normalizar_datos(X_tr, X_te) 
    w_n, b_n, hist_norm = gradient_descent(X_tr_norm, y_tr, X_te_norm, y_te, alpha=1e-3, iterations=1000) 
 
# ---------- visualización ---------- 
    steps = np.arange(0, 1000, 10).tolist() 
    if 999 not in steps: 
        steps.append(999)  # incluir última iteración 
    
    plt.figure(figsize=(16, 10)) 
    
    # Pérdida 
    plt.subplot(2, 2, 1) 
    plt.plot(hist['train_loss'], label="Loss (sin normalizar)", color='red') 
    plt.plot(hist_norm['train_loss'], label="Loss (normalizado)", color='blue') 
    plt.title("Pérdida - Entrenamiento") 
    plt.xlabel("Iteraciones (cada 10)") 
    plt.ylabel("Loss") 
    plt.legend() 
    plt.grid(True) 
    
    # Exactitud entrenamiento 
    plt.subplot(2, 2, 2) 
    plt.plot(hist['train_acc'], label="Accuracy Train (sin normalizar)", color='red') 
    plt.plot(hist_norm['train_acc'], label="Accuracy Train (normalizado)", color='blue') 
    plt.title("Exactitud - Entrenamiento") 
    plt.xlabel("Iteraciones (cada 10)") 
    plt.ylabel("Accuracy") 
    plt.ylim(0, 1) 
    plt.legend() 
    plt.grid(True) 
    
    # Exactitud test 
    plt.subplot(2, 2, 3) 
    plt.plot(hist['test_acc'], label="Accuracy Test (sin normalizar)", color='red') 
    plt.plot(hist_norm['test_acc'], label="Accuracy Test (normalizado)", color='blue') 
    plt.title("Exactitud - Test") 
    plt.xlabel("Iteraciones (cada 10)") 
    plt.ylabel("Accuracy") 
    plt.ylim(0, 1) 
    plt.legend() 
    plt.grid(True) 
    
    plt.tight_layout() 
    plt.show() 
    
    # Últimos valores para el informe 
    print("\n--- Comparación final ---") 
    print(f"Sin normalizar - Test Accuracy final: {hist['test_acc'][-1]:.4f}") 
    print(f"Normalizado     - Test Accuracy final: {hist_norm['test_acc'][-1]:.4f}") 
    
if __name__ == "__main__": main() 