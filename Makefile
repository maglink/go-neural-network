run-example:
	go build -o ./examples/simple/simple_example ./examples/simple/main.go
	./examples/simple/simple_example

run-example-2:
	go build -o ./examples/web-explorer/web_explorer ./examples/web-explorer/main.go
	./examples/web-explorer/web_explorer