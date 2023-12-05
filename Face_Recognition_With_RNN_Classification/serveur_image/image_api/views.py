from django.shortcuts import render

# Create your views here.


from django.http import HttpResponse

def upload_image(request):
	if request.method == 'GET' and request.FILES.get('image'):
		image = request.FILES['image']
		with open('phone_camera_image/student') as file:
			file.write(image.read())
		return HttpResponse('Image Upload successfully')	
	return HttpResponse('Invalid request')