# NYC-Twitter-Activities

Based on the dataset of twitter, which contains
* timestamp
* zipcode
* user id
* tweet id

We build different networks, for example
* naive network: \<zip1> \<zip2> \<number of visitors who visit both places>
* probabilistic network: \<zip1> \<zip2> \<link weight defined by a probabilistic method>

Those networks are stored in
* Network 1 for naive
* Network 2 for probabilistic
