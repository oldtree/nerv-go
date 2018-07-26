package main

import (
	"os"
	"os/signal"
	"syscall"

	nerv "github.com/oldtree/nerv-go/pkg/bayes"
	log "github.com/sirupsen/logrus"
)

func main() {
	log.Infof("アスカ・ラングレー")
	nerv.BayesLearn()
	sc := make(chan os.Signal, 1)
	signal.Notify(sc,
		syscall.SIGINT,
		syscall.SIGTERM,
		syscall.SIGQUIT,
	)
	<-sc
}
