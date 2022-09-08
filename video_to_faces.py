from models.mobilefacenet import MobileFaceNetEncoder

def video_to_faces():
    model = MobileFaceNetEncoder('cpu')

if __name__ == "__main__":
    video_to_faces()