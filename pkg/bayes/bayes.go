package bayes

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"

	"github.com/jbrukh/bayesian"
	log "github.com/sirupsen/logrus"
)

const (
	Good bayesian.Class = "Good"
	Bad  bayesian.Class = "Bad"
)

const (
	// MultinomialTf is model where frequency of token affects posterior probability
	MultinomialTf Model = 1

	// MultinomialBoolean is model like TF, but each token only calculated once for each document
	MultinomialBoolean Model = 2
)

// Model is alias of int, representing Naive-Bayes model that used in classifier
type Model int

// Class is alias of string, representing class of a document
type Class string

// Document is a group of tokens with certain class
type Document struct {
	Class  Class
	Tokens []string
}

// NewDocument return new Document
func NewDocument(class Class, tokens ...string) Document {
	return Document{
		Class:  class,
		Tokens: tokens,
	}
}

// Classifier is object for classifying document
type Classifier struct {
	Model              Model
	LearningResults    map[string]map[Class]int
	PriorProbabilities map[Class]float64
	NDocumentByClass   map[Class]int
	NFrequencyByClass  map[Class]int
	NAllDocument       int
}

// NewClassifier returns new Classifier
func NewClassifier(model Model) Classifier {
	return Classifier{
		Model:              model,
		LearningResults:    make(map[string]map[Class]int),
		PriorProbabilities: make(map[Class]float64),
		NDocumentByClass:   make(map[Class]int),
		NFrequencyByClass:  make(map[Class]int),
	}
}

// NewClassifierFromFile returns new Classifier with configuration loaded from file in path
func NewClassifierFromFile(path string) (Classifier, error) {
	classifier := Classifier{}

	fl, err := os.Open(path)
	if err != nil {
		return classifier, err
	}
	defer fl.Close()

	err = gob.NewDecoder(fl).Decode(&classifier)
	if err != nil {
		return classifier, err
	}

	return classifier, err
}

// Learn executes learning process for this classifier
func (classifier *Classifier) Learn(docs ...Document) {
	log.Infof("-----------------------start Learn-----------------------")
	defer func() {
		log.Infof("-----------------------end Learn-----------------------")
	}()
	classifier.NAllDocument += len(docs)

	for _, doc := range docs {
		classifier.NDocumentByClass[doc.Class]++

		tokens := doc.Tokens
		if classifier.Model == MultinomialBoolean {
			tokens = classifier.removeDuplicate(doc.Tokens...)
		}

		for _, token := range tokens {
			classifier.NFrequencyByClass[doc.Class]++

			if _, exist := classifier.LearningResults[token]; !exist {
				classifier.LearningResults[token] = make(map[Class]int)
			}

			classifier.LearningResults[token][doc.Class]++
		}
	}

	for class, nDocument := range classifier.NDocumentByClass {
		log.Infof("class : [%s] nDocument : [%d] NAllDocument : [%d]", class, nDocument, classifier.NAllDocument)
		classifier.PriorProbabilities[class] = math.Log(float64(nDocument) / float64(classifier.NAllDocument))
	}
}

// Classify executes classifying process for tokens
func (classifier Classifier) Classify(tokens ...string) (map[Class]float64, Class, bool) {
	log.Infof("-----------------------start Classify-----------------------")
	defer func() {
		log.Infof("-----------------------end Classify-----------------------")
	}()
	nVocabulary := len(classifier.LearningResults)
	log.Infof("learning result learn : [%d] ", nVocabulary)
	posteriorProbabilities := make(map[Class]float64)

	for class, priorProb := range classifier.PriorProbabilities {
		posteriorProbabilities[class] = priorProb
	}

	if classifier.Model == MultinomialBoolean {
		tokens = classifier.removeDuplicate(tokens...)
	}

	for class, freqByClass := range classifier.NFrequencyByClass {
		for _, token := range tokens {
			nToken := classifier.LearningResults[token][class]
			log.Infof("class : [%s] token : [%s] nToken : [%d] freqByClass : [%d]", class, token, nToken+1, freqByClass+nVocabulary)
			posteriorProbabilities[class] += math.Log(float64(nToken+1) / float64(freqByClass+nVocabulary))
		}
	}

	var certain bool
	var bestClass Class
	var highestProb float64
	for class, prob := range posteriorProbabilities {
		if highestProb == 0 || prob > highestProb {
			certain = true
			bestClass = class
			highestProb = prob
		} else if prob == highestProb {
			certain = false
		}
	}

	return posteriorProbabilities, bestClass, certain
}

// SaveClassifierToFile saves Classifier config to file in path
func (classifier Classifier) SaveClassifierToFile(path string) error {
	fl, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fl.Close()

	err = gob.NewEncoder(fl).Encode(&classifier)
	if err != nil {
		return err
	}

	return nil
}

func (classifier *Classifier) removeDuplicate(tokens ...string) []string {
	mapTokens := make(map[string]struct{})
	newTokens := []string{}

	for _, token := range tokens {
		mapTokens[token] = struct{}{}
	}

	for key := range mapTokens {
		newTokens = append(newTokens, key)
	}

	return newTokens
}

// func BayesLearn() {
// 	classifier := bayesian.NewClassifier(Good, Bad)
// 	goodStuff := []string{"tall", "rich", "handsome"}
// 	badStuff := []string{"poor", "smelly", "ugly"}
// 	classifier.Learn(goodStuff, Good)
// 	classifier.Learn(badStuff, Bad)
// 	//classifier.ConvertTermsFreqToTfIdf()
// 	scores, likely, _ := classifier.LogScores([]string{"tall", "rich", "handsome"})
// 	fmt.Printf("scores : %v ,likely : %d \n", scores, likely)
// 	probs, likely, _ := classifier.ProbScores([]string{"tall"})
// 	fmt.Printf("probs : %v ,likely : %d \n", probs, likely)
// }

func BayesImpl() {
	classifier := NewClassifier(1)
	goodStuff := []string{"tall", "rich", "handsome"}
	badStuff := []string{"poor", "smelly", "ugly"}
	classifier.Learn([]Document{Document{"GOOD", goodStuff}}...)
	classifier.Learn([]Document{Document{"BAD", badStuff}}...)
	//classifier.ConvertTermsFreqToTfIdf()
	probeMap, classType, certain := classifier.Classify([]string{"tall", "rich", "handsome"}...)

	fmt.Printf("probeMap : %v ,classType : %s certain : %t \n", int(probeMap["GOOD"]), classType, certain)
}
