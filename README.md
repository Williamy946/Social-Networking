# Social-Networking

## Data background

- The dataset comes from the famous online activity organizing website Meetup, the mechanism is that:

- The whole meetup community is consist of a number of groups, each group owns a number of users. Users are free to enter or quit any group.

- The activities are started up by a number of group members (some organizers' data is lost), and only group members are invited.

- Group members choose whether or not to participate in activities (Yes/No/Maybe), but not all members will respond.

- Some activities may limit the headcount but not constraints.
- The specific experiment data includes Group-Topic, Member-Topic, and Group-Event.

## Experiment Purpose

For the given datasets, predict the user response to activity invitations. (Yes/No/Maybe)

## Data processing

For an unweighted social graph, we build the social network through the Group-User data. If any of the two users are in the same group, there is an edge between them. Since the social network is too large to train, we random samples 8% of edges in the whole social network for prediction.  

For a weighted social graph, we compute the edge weights according to the sharing group, events, and common topics of different users.

Meanwhile, since the timestamp of events is large integer, we standardize them into float numbers with $\delta = 1, \sigma=0$. Moreover, the limits of headcount are mostly ranging from 0 to 100, with a small fraction of outlier values 9999, 10000, etc. We set the maximum value equal to 100 and standardize them.

## Prediction Model

We learn the user embedding and group embedding by graph neural networks. Then, we construct the embedding of events based on the embedding of groups, the embedding of activity organizers, the timestamp of events and the limited headcounts. After that, we compute the user-event interaction probabilites by multiplying the user embeddings with the event embeddings. 

We use 5-folds cross validation and take MSE as loss function.

The results are as follows:

**Unweighted Social Networks**

|       | precision | recall | F1     |
| ----- | --------- | ------ | ------ |
| No    | 0.6942    | 0.6714 | 0.6825 |
| Maybe | 0         | 0      | 0      |
| Yes   | 0.7560    | 0.7859 | 0.7706 |

The Final Accuracy is 0.7308.

**Weighted Social Networks**

|       | precision | recall | F1     |
| ----- | --------- | ------ | ------ |
| No    | 0.6994    | 0.6506 | 0.7770 |
| Maybe | 0         | 0      | 0      |
| Yes   | 0.7535    | 0.8022 | 0.7770 |

The Final Accuracy is 0.7326