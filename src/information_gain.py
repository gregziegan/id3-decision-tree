import math


def info_gain(examples, attribute, entropy_of_set):
   gain = entropy_of_set
   for value in attributeValues(examples, attribute):
       sub = subset(examples, attribute, value)
       gain -=  (number in sub)/len(examples) * entropy(sub)
   return gain


def entropy(examples, target_attribute):
   result = 0
   target_examples = summarize_examples(examples, target_attribute)
   for example in target_examples:
       proportion = example/len(examples)
       result -= proportion * math.log(proportion, 2)
   return result


def summarize_examples(examples, target_attribute):
    pass