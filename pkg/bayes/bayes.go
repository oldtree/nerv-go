package bayes

import "github.com/jbrukh/bayesian"
import "fmt"

const (
	Good bayesian.Class = "Good"
	Bad  bayesian.Class = "Bad"
)

func BayesLearn() {
	classifier := bayesian.NewClassifier(Good, Bad)
	goodStuff := []string{"tall", "rich", "handsome"}
	badStuff := []string{"poor", "smelly", "ugly"}
	classifier.Learn(goodStuff, Good)
	classifier.Learn(badStuff, Bad)
	//classifier.ConvertTermsFreqToTfIdf()
	scores, likely, _ := classifier.LogScores([]string{"tall", "rich", "handsome"})
	fmt.Printf("scores : %v ,likely : %d \n", scores, likely)
	probs, likely, _ := classifier.ProbScores([]string{"tall", "girl"})
	fmt.Printf("probs : %v ,likely : %d \n", probs, likely)
}
