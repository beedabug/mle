# build the image
docker build -t glm -f Dockerfile .

# run
docker run -p 1313:1313 glm
