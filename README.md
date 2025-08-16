### Research Theme
This study compares Chinese and American netizens’ attention and emotional changes toward the COVID-19 pandemic. 
The datasets include Weibo data, Twitter data, and pandemic statistics from both countries. 
The research consists of two main parts: **sentiment classification** and **topic clustering**.

<img width="600" alt="image" src="https://github.com/user-attachments/assets/812682dc-b593-40c1-8bb7-e0ff6b9e7f9a" />

### Sentiment Classification

**1. Twitter**: We conducted both five-class and three-class sentiment classification on 40,000 tweets, testing Decision Tree, Naïve Bayes, SVM, BERT, and neural network models, with One-vs-One and One-vs-Rest strategies.
   The final model selected was a three-class SVM with a polynomial kernel, achieving 69% accuracy.

**2. Weibo**: A LinearSVC binary classifier was used, achieving 89% accuracy.


### Topic Clustering

**1. Twitter**: Using both 1% and 10% of the dataset, we experimented with LDA and Mallet-optimized LDA models for 2–50 topics.
   The optimal Mallet model clustered 100,000 tweets into ten major topics—study and travel, home quarantine, protection, government, international pandemic, etc.—with a coherence score of 0.45.

**2. Weibo**: Using 10% of the dataset, the Mallet model produced five clusters—frontline medical staff, daily life, prevention measures, global pandemic, and domestic pandemic—with a coherence score of 0.63.


### Comparative Analysis of Public Opinion
Both Chinese and American online public opinion were divided into three stages:

- **China (Weibo)**:

  - **Stage 1 (Jan 19–Feb 12)**: Following the Jan 20 news confirming human-to-human transmission, Weibo activity increased, dominated by prayers and concern for Wuhan.
  
  - **Stage 2 (Feb 12-Mar 7)**: Confirmed cases surged due to updated diagnostic criteria, leading to a spike in discussions.
  
  - **Stage 3 (after Mar 7)**: As the domestic situation came under control, discussions shifted toward honoring medical staff, withdrawal of medical teams, and the global situation (Italy, U.S., U.K., etc.).
  
  - **Sentiment**: Generally positive throughout, with brief negative spikes during the outbreak in Wuhan, the lockdown lifting, and April 10 (mainly due to “global pandemic” topics).

- **U.S. (Twitter)**:

  - Stage 1 (Feb 1–Feb 22): Early stage, with low domestic concern and focus on international cases (frequent mentions of China).
  
  - Stage 2 (Feb 23–Mar 11): Rising concern as cases surged, states declared emergencies, and the first U.S. death occurred. Discussions centered on government and politics (keywords: “Trump,” “president”).
  
  - Stage 3 (after Mar 12): Following Trump’s Mar 13 national emergency declaration, discussions increased on “study and travel” and “home quarantine” as schools closed and restrictions grew (keywords: “cancel,” “work,” “lockdown”).
  
  - Sentiment: Initially negative, then gradually positive and stable. Early negativity was driven by “international pandemic” and “U.S. government” topics, but as government measures were introduced, sentiment improved.


### Conclusions

**1. Attention**:

- Chinese netizens engaged earlier, focusing on domestic epidemic trends, frontline efforts, and collective support.

- American netizens engaged later, with stronger attention to personal life, work, and government responses.

- Both groups paid more attention to the global situation when domestic outbreaks were less severe.

**2. Emotion**:

- Chinese netizens maintained a predominantly positive and stable sentiment, with limited fluctuations.

- American netizens shifted from negative in the early stages to more positive and stable later, showing a clear correlation between government interventions and public sentiment.
