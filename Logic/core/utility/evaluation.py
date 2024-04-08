from typing import List
import numpy as np
import wandb

class Evaluation:

    def __init__(self, name: str):
            # wandb.init('IMBD_IR_System', 'Asemaeneh')
            self.name = name
    def precision_by_quary(self, actual: List[str], predicted: List[str]):
        precision = 0.0
        if len(predicted) == 0:
            return 0.0
        num_relevant = 0
        for item in predicted:
            if item in actual:
                num_relevant += 1
        precision = num_relevant / len(predicted)
        return precision
    
    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        #TODO: flat or mean?
        precision = []   
        for i, query in enumerate(predicted):
            precision.append(self.precision_by_quary(actual[i], query))
        return np.mean(precision)
        # flat_actual = [item for sublist in actual for item in sublist]
        # flat_predicted = [item for sublist in predicted for item in sublist]
        # return self.precision_by_quary(flat_actual, flat_predicted)
        
    def recal_by_quary(self, actual: List[str], predicted: List[str]):
        recall = 0.0
        if len(actual) == 0:
            return 0.0

        num_relevant = 0
        for item in predicted:
            if item in actual:
                num_relevant += 1

        recall = num_relevant / len(actual)
        return recall
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recals = []   
        for i, query in enumerate(predicted):
            recals.append(self.recal_by_quary(actual[i], query))
        return np.mean(recals)    
        # flat_actual = [item for sublist in actual for item in sublist]
        # flat_predicted = [item for sublist in predicted for item in sublist]
        # return self.recal_by_quary(flat_actual, flat_predicted)
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def AP_by_quary(self, actual: List[str], predicted: List[str]):
        AP = 0.0
        num_correct = 0
        for i, item in enumerate(predicted):
            if item in actual:
                num_correct += 1
                AP += num_correct / (i + 1)
        if num_correct == 0:
            return 0.0
        AP /= num_correct
        return AP
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        flat_actual = [item for sublist in actual for item in sublist]
        flat_predicted = [item for sublist in predicted for item in sublist]
        return self.AP_by_quary(flat_actual, flat_predicted)
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        APs = []   
        for i, query in enumerate(predicted):
            APs.append(self.recal_by_quary(actual[i], query))
        return np.mean(APs)    
    
    def DCG_by_quary(self, actual: List[str], predicted: List[str])->float:
        DCG = 0.0
        for i, item in enumerate(predicted):
            if item in actual:
                if(i == 0):
                    DCG += 1
                else:
                    DCG += 1 / np.log2(i + 1)
        return DCG
    
    def calculate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[str]
            The actual results
        predicted : List[str]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        flat_actual = [item for sublist in actual for item in sublist]
        flat_predicted = [item for sublist in predicted for item in sublist]
        return self.DCG_by_quary(flat_actual, flat_predicted)
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCGs = []
        for i, query in enumerate(predicted):
            DCG = self.DCG_by_quary(actual[i], query)
            ideal_DCG =  self.DCG_by_quary(actual[i], actual[i])
            if ideal_DCG == 0:
                NDCGs.append(0.0)
            NDCGs.append(DCG / ideal_DCG)            
        return np.mean(NDCGs)
        
    def RR_by_quary(self, actual: List[str], predicted: List[str]) -> float:
        RR = 0.0
        for i,predict in enumerate(predicted):
            if predict in actual:
                RR = 1/(i + 1)
                break
        return RR
    
    def calculate_RR(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        flat_actual = [item for sublist in actual for item in sublist]
        flat_predicted = [item for sublist in predicted for item in sublist]
        return self.RR_by_quary(flat_actual, flat_predicted)
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        # total_RR = sum(self.calculate_RR([a], [p]) for a, p in zip(actual, predicted))
        # MRR = total_RR / len(actual)
        # return MRR
        MRRs = []   
        for i, query in enumerate(predicted):
            MRRs.append(self.RR_by_quary(actual[i], query))
        return np.mean(MRRs)   
    

    def print_evaluation(self,precision, recall, f1, map, ndcg, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        #print(f"Average Precision: {ap}")
        print(f"Mean Average Precision: {map}")
        #print(f"Discounted Cumulative Gain: {dcg}")
        print(f"Normalized Discounted Cumulative Gain: {ndcg}")
        #print(f"Reciprocal Rank: {rr}")
        print(f"Mean Reciprocal Rank: {mrr}")
      

    def log_evaluation(self,precision, recall, f1, map, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        #Log the evaluation metrics using Wandb
        wandb.log({
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        #"Average Precision": ap,
        "Mean Average Precision": map,
        #"Discounted Cumulative Gain": dcg,
        "Normalized Discounted Cumulative Gain": ndcg,
        #"Reciprocal Rank": rr,
        "Mean Reciprocal Rank": mrr
        })

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        #ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        #dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        #rr = self.calculate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, map_score, ndcg, mrr)
        #self.log_evaluation(precision, recall, f1, map_score, ndcg, mrr)


eval = Evaluation('test')
eval.calculate_evaluation([['batman','the batman']],[['dark knight','batman']])
