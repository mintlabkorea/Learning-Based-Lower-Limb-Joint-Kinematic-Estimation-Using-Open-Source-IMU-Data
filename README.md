# Learning Based Lower Limb Joint Kinematic Estimation Using Open Source IMU Data

**Authors**: Benjamin Hur, Sunin Baek, Inseung Kang*, and Daekyum Kim*
 



## Abstract
This study introduces a deep learning framework for estimating lower-limb joint kinematics using inertial measurement units (IMUs). While deep learning methods avoid sensor drift, extensive calibration, and complex setup procedures, they require substantial data. To meet this demand, we leveraged an open-source dataset to develop and evaluate three training approaches. The first involved training a model exclusively on data from a single user, resulting in high accuracy for that individual only. The second approach trained a model on data from multiple users to generalize across individuals; however, demonstrated lower accuracy due to variations in gait patterns across users. The third approach added transfer learning to the second, improving estimation accuracy for new users through fine-tuning with a small portion of their data. This model overcame the limitations of the previous methods and achieved comparable performance to inverse kinematics, making it an effective solution for diverse populations. Additionally, our analysis on IMU combinations suggests that IMUs placed on the femur and calcaneus are the best for most cases. This framework not only reduces the need for extensive data collection but also enhances personalized gait analysis, enabling more efficient and accessible applications in both clinical assessments and real-world environments for broader use.



