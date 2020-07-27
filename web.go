package gnn

import (
	"github.com/gorilla/websocket"
	"net/http"
)

func (nn *NeuralNetwork) ListenAddr(addr string) error {
	var upgrader = websocket.Upgrader{}
	http.HandleFunc("/network", func(w http.ResponseWriter, r *http.Request) {
		conn, _ := upgrader.Upgrade(w, r, nil)
		nn.onUpdate = func() {
			conn.WriteJSON(nn)
		}
		nn.onUpdate()
		conn.SetCloseHandler(func(code int, text string) error {
			nn.onUpdate = nil
			return nil
		})
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				return
			}
		}
	})
	http.Handle("/", http.FileServer(http.Dir("./web")))
	return http.ListenAndServe(addr, nil)
}
