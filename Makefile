build:
	docker build -t crae .
shell:
	docker run -it --rm crae
notebook:
	docker run -it -p 9999:9999 --rm -v ${PWD}/data:/data -v ${PWD}/src:/srv -w /experiments \
	-v ${PWD}/experiments:/experiments crae jupyter-notebook --allow-root --ip='*' --port=9999 \
	--NotebookApp.password='' --NotebookApp.token=''
