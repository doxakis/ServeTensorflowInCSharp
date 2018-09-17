# ServeTensorflowInCSharp
Let's consider the following image:

<p align="center">
    <img alt="Example" src="files/cat1.jpg" />
</p>

Let's compare the results of the model Inception V3 if we use TensorFlowSharp (c#) instead of TensorFlow (python).

| Label              | TensorFlow         | TensorFlowSharp     |
|--------------------|--------------------|---------------------|
| dummy              | 0.0000368240507669 | 0.00003381444000000 |
| kit fox            | 0.0001495147880632 | 0.00014453140000000 |
| English setter     | 0.0000872057134984 | 0.00008187198000000 |
| Siberian husky     | 0.0000672979585943 | 0.00006730104000000 |
| Australian terrier | 0.0000622533771093 | 0.00005606212000000 |
| English springer   | 0.0000364902625734 | 0.00003416507000000 |
| grey whale         | 0.0000557607381779 | 0.00004978874000000 |
| lesser panda       | 0.0000176537014340 | 0.00001699535000000 |
| Egyptian cat       | 0.8147186636924744 | 0.84351500000000000 |
| ibex               | 0.0000344633881468 | 0.00003263445000000 |
| Persian cat        | 0.0002733957662713 | 0.00023066850000000 |
| cougar             | 0.0000940313548199 | 0.00009165877000000 |
| gazelle            | 0.0000283752888208 | 0.00002851867000000 |
| porcupine          | 0.0000832021178212 | 0.00007991350000000 |
| ...                | ...                | ...                 |

Well, it turn out they both tell us it's an egyptian cat.
Further work is needed to find the reasons there is a discrepancy in the results.

# Copyright and license
Code released under the MIT license.