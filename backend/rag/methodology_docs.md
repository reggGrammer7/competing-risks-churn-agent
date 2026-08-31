# Doc: what-is-a-cif
A cumulative incidence function (CIF) gives the probability that a specific
competing event has occurred by time t, accounting for the fact that a
customer who experiences a DIFFERENT event first is removed from the risk
set for this one. It is not the same as 1 minus a Kaplan-Meier survival
curve for that cause alone -- that naive calculation overstates the true
risk of any single cause.

# Doc: why-km-overstates-risk
Kaplan-Meier treats every non-target-event as a censoring, including
customers who left for a DIFFERENT reason. This wrongly keeps them in the
risk set for the cause being studied when they've actually been removed
from it entirely. The Aalen-Johansen estimator corrects for this by only
counting customers still genuinely at risk of the specific cause.

# Doc: cause-specific-cox-vs-fine-gray
A cause-specific Cox model estimates how covariates affect the INSTANTANEOUS
hazard of one cause, treating other causes as censoring. It answers "among
those still at risk, how does this covariate change the rate of this
specific event right now." Fine-Gray instead models the effect directly on
the cumulative incidence, answering "how does this covariate change the
actual probability of this outcome by time t" -- a more decision-relevant
quantity for a business action, though it does not have as clean a
biological/mechanistic interpretation as the cause-specific hazard.

# Doc: what-the-strawman-misses
The XGBoost binary classifier ignores time-to-event structure entirely. It
treats a customer who joined last week and hasn't churned yet identically
to a five-year loyal customer -- both are just "negative" -- which biases
the model toward patterns visible in longer-tenured customers. It also
collapses all churn causes into one outcome, so it cannot say WHICH reason
is driving risk or WHEN the risk is highest.

# Doc: what-the-collapsed-causes-mean
Churn reasons are grouped into four buckets: Price sensitivity (cost-driven
departures), Dissatisfaction (service/support quality issues), Competitive
loss (a competitor's offer), and Non-behavioral (moves, and other reasons
unrelated to the relationship with the company). This grouping was chosen
so each bucket has enough observed events to fit a stable model.
