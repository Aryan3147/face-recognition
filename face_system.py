import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import pickle


class FaceDatabase:
    def __init__(self, database_dir="face_database"):
        self.database_dir = Path(database_dir)
        self.encodings_file = self.database_dir / "encodings.pkl"
        self.images_dir = self.database_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.encodings = {}
        self.load()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )

    def load(self):
        if self.encodings_file.exists():
            with open(self.encodings_file, 'rb') as f:
                self.encodings = pickle.load(f)

    def save(self):
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.encodings, f)

    def preprocess_face(self, face_image):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        return hist.flatten()

    def register_from_frame(self, name, face_image):
        embedding = self.preprocess_face(face_image)
        
        self.encodings[name] = embedding
        dest_path = self.images_dir / f"{name}.jpg"
        cv2.imwrite(str(dest_path), face_image)
        self.save()
        return True

    def recognize(self, face_embedding):
        if not self.encodings:
            return None, 0.0
        
        known_embeddings = list(self.encodings.values())
        names = list(self.encodings.keys())
        
        similarities = []
        for known in known_embeddings:
            hist1 = known.flatten() / (np.sum(known) + 1e-6)
            hist2 = face_embedding.flatten() / (np.sum(face_embedding) + 1e-6)
            similarity = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] > 0.5:
            return names[best_match_idx], similarities[best_match_idx]
        
        return None, 0.0

    def remove_person(self, name):
        if name in self.encodings:
            del self.encodings[name]
            image_path = self.images_dir / f"{name}.jpg"
            if image_path.exists():
                image_path.unlink()
            self.save()
            return True
        return False

    def get_all_names(self):
        return list(self.encodings.keys())

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) >= 1:
            x, y, w, h = faces[0]
            return x, y, w, h
        
        faces = self.profile_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) >= 1:
            x, y, w, h = faces[0]
            return x, y, w, h
        
        return None


class FaceRegistrationSystem:
    def __init__(self, database_dir="face_database"):
        self.db = FaceDatabase(database_dir)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, 1280)
        self.capture.set(4, 720)
        
    def register_new_face(self, name):
        print(f"\n=== Registering: {name} ===")
        print("Look at the camera...")
        
        registered = False
        countdown = 30
        
        while not registered and countdown > 0:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            display = frame.copy()
            cv2.putText(display, f"Registering: {name}", (10, 40),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
            
            face_rect = self.db.detect_face(frame)
            
            if face_rect:
                x, y, w, h = face_rect
                padding = 20
                x_p = max(0, x - padding)
                y_p = max(0, y - padding)
                w_p = w + padding * 2
                h_p = h + padding * 2
                cv2.rectangle(display, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 255, 0), 3)
                countdown -= 1
                cv2.putText(display, f"Capturing in: {countdown//10 + 1}", (x_p, y_p - 10),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
                
                if countdown <= 0:
                    face_image = frame[y:y+h, x:x+w]
                    self.db.register_from_frame(name, face_image)
                    print(f"Successfully registered: {name}")
                    registered = True
            else:
                cv2.putText(display, "No face detected - align your face", (150, 400),
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
            
            cv2.imshow('Face Registration', display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow('Face Registration')
        return registered

    def run_detection(self):
        print("\n=== Starting Real-time Face Detection ===")
        print("Press 'q' to quit")
        print("Press 'r' to register a new face")
        
        log_file = open("attendance.log", "a")
        
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            
            display = frame.copy()
            
            face_rect = self.db.detect_face(frame)
            
            if face_rect:
                x, y, w, h = face_rect
                padding = 20
                x_p = max(0, x - padding)
                y_p = max(0, y - padding)
                w_p = w + padding * 2
                h_p = h + padding * 2
                face_image = frame[y:y+h, x:x+w]
                embedding = self.db.preprocess_face(face_image)
                
                name, confidence = self.db.recognize(embedding)
                
                if name:
                    color = (0, 255, 0)
                    label = f"{name}"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{name},{timestamp}\n")
                    log_file.flush()
                else:
                    color = (0, 0, 255)
                    label = "Unknown"
                
                cv2.rectangle(display, (x_p, y_p), (x_p + w_p, y_p + h_p), color, 3)
                cv2.rectangle(display, (x_p, y_p + h_p - 35), (x_p + w_p, y_p + h_p), color, cv2.FILLED)
                cv2.putText(display, label, (x_p + 10, y_p + h_p - 10),
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display, f"Registered: {len(self.db.get_all_names())}", 
                       (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, "Press 'r' to register | 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 180, 180), 1)
            
            cv2.imshow('Face Detection', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cv2.destroyWindow('Face Detection')
                name = input("Enter name to register: ").strip()
                if name:
                    self.register_new_face(name)
        
        log_file.close()
        self.capture.release()
        cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("  Real-Time Face Registration & Detection System")
    print("=" * 50)
    print("\n1. Start Detection")
    print("2. Register New Face")
    print("3. List Registered Faces")
    print("4. Remove a Face")
    print("5. Exit")
    
    system = FaceRegistrationSystem()
    
    while True:
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            cv2.namedWindow('Face Detection')
            system.run_detection()
        elif choice == '2':
            name = input("Enter name to register: ").strip()
            if name:
                system.register_new_face(name)
        elif choice == '3':
            names = system.db.get_all_names()
            print(f"\nRegistered faces ({len(names)}):")
            for name in names:
                print(f"  - {name}")
        elif choice == '4':
            name = input("Enter name to remove: ").strip()
            if name and system.db.remove_person(name):
                print(f"Removed: {name}")
            else:
                print("Name not found")
        elif choice == '5':
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main()