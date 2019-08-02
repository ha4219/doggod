from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"강아지", "limit":1000, "print_urls": True, "chromedriver":'chromedriver'}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)

'''
    해야하는 작업
    1. jpg로 형식 맞추기
    2. file 이름 맞추기
    에시 0001.jpg
'''