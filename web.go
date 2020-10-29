package gnn

import (
	"github.com/gorilla/websocket"
	"net/http"
	"time"
)

const sendUpdatesFrequency = 3 * time.Second

func (nn *NeuralNetwork) ListenAddr(addr string) error {
	var upgrader = websocket.Upgrader{}
	http.HandleFunc("/network", func(w http.ResponseWriter, r *http.Request) {
		conn, _ := upgrader.Upgrade(w, r, nil)
		conn.WriteJSON(nn)
		ticker := time.NewTicker(sendUpdatesFrequency)
		done := make(chan bool)
		go func() {
			for {
				select {
				case <-done:
					ticker.Stop()
					return
				case <-ticker.C:
					conn.WriteJSON(nn)
				}
			}
		}()
		conn.SetCloseHandler(func(code int, text string) error {
			close(done)
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
