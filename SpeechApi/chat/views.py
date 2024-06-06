from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes,parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from django.conf import settings
from rest_framework import status
from django.core.cache import cache
import pickle
import os
import uuid
from .models import AudioFile
from .audioprocessing import getPrediction




class LoginAPIView(APIView):
    def post(self, request):
        email = request.data.get("email")
        password = request.data.get("password")

        # Perform authentication
        user = User.objects.filter(email=email).first()
        if user is None or not user.check_password(password):
            return Response(
                {"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED
            )

        # Assuming authentication is successful, generate the token
        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        return Response({"access_token": access_token})


class RegistrationAPIView(APIView):
    def post(self, request):
        email = request.data.get("email")
        name = request.data.get("name")
        password = request.data.get("password")

        # Check if the email is already registered
        if User.objects.filter(email=email).exists():
            return Response(
                {"error": "Email already registered"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Create a new user
        user = User.objects.create_user(username=email, email=email, password=password)
        user.first_name = name
        user.save()

        # Generate the token for the newly registered user
        refresh = RefreshToken.for_user(user)
        access_token = str(refresh.access_token)
        return Response({"access_token": access_token})
    
@api_view(['POST'])
def upload_audio(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        # Generate a unique ID for the audio file
        unique_id = uuid.uuid4()

        # Save the audio file to the media folder
        file_name = str(unique_id) + '_' + audio_file.name
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        with open(file_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # Extract the name of the audio file
        audio_name = audio_file.name.split('.')[0]

        # Get prediction for the uploaded audio file
        response = getPrediction(file_path)
        # print("************",response)
        if os.path.exists(file_path):
           os.remove(file_path)

        # Save the information to the database
        # AudioFile.objects.create(
        #     id=unique_id,
        #     name=audio_name,
        #     file=file_name,  # Save the relative file path
        #     is_fake=False
        # )
        return Response({'message':response}, status=status.HTTP_201_CREATED)
    else:
        return Response({'error': 'Please provide an audio file.'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def get_all_audio(request):
    if request.method == 'GET':
        audio_files = AudioFile.objects.all()
        data = []
        for audio_file in audio_files:
            data.append({
                'id': audio_file.id,
                'name': audio_file.name,
                'file': audio_file.file.url,
                'is_fake': audio_file.is_fake
            })
        return Response({"data":data})

