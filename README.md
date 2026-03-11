# MMA - X-ray Classifier API

API FastAPI pour classifier une radiographie (`NORMAL` / `FRACTURE`) avec interface web integree.

## Prerequis

- Docker Desktop (recommande)
- ou Python 3.10 si execution locale
- Fichier de modele present a la racine : `best_densenet169_legfracture.pt`

## Lancer le projet (Docker Compose)

```powershell
docker compose up --build
```

Puis ouvrir :

- Interface web : http://localhost:8000/
- Healthcheck : http://localhost:8000/health

Arreter :

```powershell
docker compose down
```

## Deploiement Kubernetes

Les manifests Kubernetes sont dans `k8s/` :

- `k8s/mma-deployment.yml`
- `k8s/mma-service.yml`

### Docker Hub + Minikube

1. Construire l'image Docker avec votre nom Docker Hub :

```powershell
docker build -t yourdockerhubusername/mma-api:v1 .
```

2. Verification optionnelle :

```powershell
docker images
```

3. Tester localement l'image :

```powershell
docker run --rm -p 8000:8000 yourdockerhubusername/mma-api:v1
```

Puis ouvrir :

- Interface web : http://localhost:8000/
- Swagger : http://localhost:8000/docs
- Healthcheck : http://localhost:8000/health

4. Pousser l'image sur Docker Hub :

```powershell
docker login
docker push yourdockerhubusername/mma-api:v1
```

5. Installer Minikube puis ajouter son executable au `PATH`.

6. Installer `kubectl` puis ajouter son executable au `PATH`.

Sous Windows, l'installation peut necessiter un terminal ouvert en mode administrateur.

7. Demarrer Minikube :

```powershell
minikube start
```

8. Selectionner le contexte Kubernetes :

```powershell
kubectl config use-context minikube
```

9. Verifier les noeuds :

```powershell
kubectl get nodes
```

10. Adapter l'image dans `k8s/mma-deployment.yml` si besoin :

```yaml
image: yourdockerhubusername/mma-api:v1
```

11. Creer le deployment :

```powershell
kubectl apply -f k8s/mma-deployment.yml
```

12. Verifier les pods :

```powershell
kubectl get pods
```

13. Creer le service :

```powershell
kubectl apply -f k8s/mma-service.yml
```

14. Verifier les services :

```powershell
kubectl get services
```

15. Ouvrir le service Minikube :

```powershell
minikube service mma-api
```

Selon le mode reseau de Minikube, utilisez ensuite l'URL affichee puis ajoutez `/docs` pour tester l'API.

16. Augmenter le nombre de replicas si besoin :

```powershell
kubectl scale deployment mma-api --replicas=5
```

### Mise a jour apres modification FastAPI

Si vous modifiez `app.py`, `frontend.py` ou `model_service.py`, reconstruisez et redeployez une nouvelle version :

```powershell
docker build -t yourdockerhubusername/mma-api:v2 .
docker push yourdockerhubusername/mma-api:v2
```

Mettez ensuite a jour l'image dans `k8s/mma-deployment.yml` :

```yaml
image: yourdockerhubusername/mma-api:v2
```

Puis relancez le deployment :

```powershell
kubectl apply -f k8s/mma-deployment.yml
kubectl rollout status deployment/mma-api
```

Vous pouvez aussi reappliquer tout le dossier :

```powershell
kubectl apply -f k8s/
```

### Acces alternatifs

Si vous ne voulez pas utiliser `minikube service`, vous pouvez aussi tester via :

- NodePort : `http://<node-ip>:30080/`
- Healthcheck : `http://<node-ip>:30080/health`

Ou avec un port-forward :

```powershell
kubectl port-forward service/mma-api 8000:80
```

Puis ouvrir :

- Interface web : http://localhost:8000/
- Swagger : http://localhost:8000/docs
- Healthcheck : http://localhost:8000/health

## Lancer en local (sans Docker)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Variable d'environnement

- `MODEL_PATH` : chemin du fichier modele (defaut: `best_densenet169_legfracture.pt`)
- `CAM_METHOD` : methode Grad-CAM par defaut (`gradcam` ou `gradcampp`)
- `CAM_TARGET_LAYER` : couche cible par defaut pour la visualisation CAM

Exemple :

```powershell
$env:MODEL_PATH="best_densenet169_legfracture.pt"
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Test API rapide (PowerShell)

```powershell
$form = @{ file = Get-Item "C:\path\to\image.jpg" }
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict?show_image=true&show_cam=true" -Form $form
```

