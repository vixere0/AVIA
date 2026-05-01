"""
AVIA - Model Eğitim Scripti
Çalıştırmadan önce: pip install torch pandas scikit-learn joblib
Veri seti: ai4i2020.csv aynı klasörde olmalı
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


print("Veri yükleniyor...")
df = pd.read_csv('ai4i2020.csv')

X = df[['Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]']].values

y = df['Machine failure'].values

print(f"   Toplam kayıt   : {len(y)}")
print(f"   Arızalı kayıt  : {y.sum()} (%{y.mean()*100:.1f})")
print(f"   Sağlıklı kayıt : {(1-y).sum()} (%{(1-y).mean()*100:.1f})")


X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled  = scaler.transform(X_test_raw)


joblib.dump(scaler, 'scaler.pkl')
print("\n scaler.pkl kaydedildi")
print(f"   Ortalamalar : {scaler.mean_.round(2)}")
print(f"   Std değerler: {scaler.scale_.round(2)}")


X_train = torch.FloatTensor(X_train_scaled)
X_test  = torch.FloatTensor(X_test_scaled)
y_train = torch.FloatTensor(y_train_raw).reshape(-1, 1)
y_test  = torch.FloatTensor(y_test_raw).reshape(-1, 1)

classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train_raw)
pos_weight = torch.tensor([class_weights[1] / class_weights[0]])
print(f"\n  pos_weight: {pos_weight.item():.2f}  (class imbalance düzeltmesi)")


class DiagnosticNet(nn.Module):
    def __init__(self):
        super(DiagnosticNet, self).__init__()
        self.layer1   = nn.Linear(5, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2   = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3   = nn.Linear(16, 8)
        self.output   = nn.Linear(8, 1)
        

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        return self.output(x)  

model = DiagnosticNet()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

epochs = 200
best_val_loss = float('inf')

for epoch in range(epochs):
   
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test).item()

       
        val_probs = torch.sigmoid(val_outputs)
        val_preds = (val_probs >= 0.5).float()
        val_acc = val_preds.eq(y_test).float().mean().item()
        true_pos  = ((val_preds == 1) & (y_test == 1)).sum().item()
        false_neg = ((val_preds == 0) & (y_test == 1)).sum().item()
        recall = true_pos / (true_pos + false_neg + 1e-8)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'diagnostic_model.pth')

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: %{val_acc*100:.1f} | "
              f"Arıza Recall: %{recall*100:.1f}")

print(f"\n En iyi model 'diagnostic_model.pth' olarak kaydedildi!")
print(f"   En iyi val loss: {best_val_loss:.4f}")

model.load_state_dict(torch.load('diagnostic_model.pth'))
model.eval()

with torch.no_grad():
    probs = torch.sigmoid(model(X_test))
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (y_test == 1)).sum().item()
    tn = ((preds == 0) & (y_test == 0)).sum().item()
    fp = ((preds == 1) & (y_test == 0)).sum().item()
    fn = ((preds == 0) & (y_test == 1)).sum().item()

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
print("\nTest Sonuçları:")
print(f"   Accuracy  : %{accuracy*100:.2f}")
print(f"   Precision : %{precision*100:.2f}")
print(f"   Recall    : %{recall*100:.2f}  ← arızaları yakalama oranı")
print(f"   F1 Score  : %{f1*100:.2f}")
print(f"\n   TP={int(tp)}, TN={int(tn)}, FP={int(fp)}, FN={int(fn)}")
print("\n Eğitim tamamlandı! 'diagnostic_model.pth' ve 'scaler.pkl' hazır.")