# 🔭 Nakshatra N-SIGHT  
### Full-Stack Spectral Analysis Platform

Nakshatra **N-SIGHT** is an industry-grade, full-stack web application designed for **astronomical spectral analysis**.  
It integrates a **Machine Learning pipeline** with **cloud-based data management**, enabling researchers and students to upload, analyze, process, and classify stellar spectral data.

🚀 **Live Demo:**  
> _Insert your Streamlit Cloud link here_

---

## 📌 Project Overview

This project demonstrates **end-to-end full-stack competency**, bridging raw scientific data collection with modern cloud-native web architecture.

### Users can:
- **Ingest Data:** Upload FITS/SER files or stream live from QHY cameras
- **Process Signals:** Apply noise reduction (Savitzky–Golay, Gaussian) and derivative spectroscopy
- **Train AI Models:** Build datasets and train Random Forest models directly in-browser
- **Persist Data:** Securely store observations using cloud NoSQL DB with RBAC

---

## ✨ Key Features

### 🔐 Identity & Security
- Custom Login & Registration System  
- Password hashing using **bcrypt (salt + hash)**
- **Role-Based Access Control (RBAC)**
  - **Student:** Access only their own spectral data
  - **Admin:** Global access to users and analytics

---

### 💾 Database & Architecture
- **Cloud Database:** Google Firebase Firestore (NoSQL)
- **CRUD Operations**
  - Create: Save spectral observations
  - Read: View & filter personal library
  - Update: Modify labels & metadata
  - Delete: Remove records permanently
- **Real-time Search & Filtering**

---

### 🧠 ML Pipeline & Spectral Analysis
- Signal Processing:
  - Noise reduction
  - Continuum normalization
  - Peak detection
- Feature Engineering:
  - Automated resampling to fixed-size vectors
- In-browser Model Training:
  - Random Forest Classifier (Scikit-learn)
- Real-time inference for star classification

---

## 🧰 Tech Stack

| Component        | Technology            | Description |
|------------------|-----------------------|------------|
| Frontend         | Streamlit             | Python-based rapid UI framework |
| Backend          | Python 3.12           | Core logic, ML, signal processing |
| Database         | Firebase Firestore    | NoSQL cloud database |
| ML Engine        | Scikit-learn          | Random Forest classifier |
| Science Libraries| Astropy, SciPy        | FITS handling & signal filtering |
| Visualization    | Plotly                | Interactive spectral plots |
| Security         | Bcrypt                | Secure password hashing |

---

## 🏗️ System Architecture

The app follows a **Monolithic Cloud Architecture**, where frontend and backend logic reside in a single Streamlit container.

```mermaid
graph TD
    User[User / Client] -->|HTTPS| App[Streamlit Application]

    subgraph "Application Logic (Python)"
        App --> Auth[Auth Module (Bcrypt)]
        App --> ML[ML Engine (Scikit-Learn)]
        App --> SP[Signal Processor (SciPy/Astropy)]
    end

    subgraph "Cloud Data Layer"
        Auth -->|Read/Write| DB[(Firebase Firestore)]
        App -->|CRUD Ops| DB
    end

    subgraph "Hardware Layer"
        User -->|USB Stream| Camera[QHY/Webcam]
    end
