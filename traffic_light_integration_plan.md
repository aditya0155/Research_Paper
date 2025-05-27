# Plan for Integrating Traffic Light Awareness into CARLA RL Environment

This document outlines the plan to enhance `carla_env.py` by integrating traffic light awareness and modifying reward structures.

**I. Understanding the Existing Code Structure**

The key areas for modification in `carla_env.py` are:

*   **`__init__(self, ...)`:** For initializing new parameters (e.g., reward for waiting at red light) and state variables (e.g., current traffic light state). Potentially adjust `self.observation_space`.
*   **`reset(self)`:** To reset new traffic light-related state variables at the start of each episode.
*   **`step(self, action)`:** The core method for:
    *   Fetching the current traffic light state.
    *   Calculating the new positive reward for waiting at a red light.
    *   Modifying the "idle penalty" logic to suspend it when waiting at a red light.

**II. Detailed Plan for Modifications**

1.  **Add New Instance Variables and Parameters:**
    *   In the `__init__` method:
        *   `self.current_traffic_light_state = None`: To store the state (e.g., `carla.TrafficLightState.RED`, `GREEN`) of the traffic light affecting the ego vehicle.
        *   `reward_wait_red_light_value` (e.g., default `5.0`): Add as a new parameter to `__init__`, stored as `self.reward_wait_red_light`.
        *   `self.is_waiting_at_red_light = False`: A boolean flag for logic simplification.

2.  **Modify `__init__` method:**
    *   Accept `reward_wait_red_light_value` and initialize `self.reward_wait_red_light`.
    *   Initialize `self.current_traffic_light_state = None`.
    *   Initialize `self.is_waiting_at_red_light = False`.
    *   *(Decision: For now, we will not add traffic light state directly to the agent's observation vector to keep the initial changes focused. This can be a future enhancement.)*

3.  **Modify `reset` method:**
    *   Add `self.current_traffic_light_state = None`.
    *   Add `self.is_waiting_at_red_light = False`.

4.  **Modify `step` method:**

    *   **A. Determine Traffic Light State and Waiting Condition:**
        *   Early in `step`, after getting vehicle state, use `self.vehicle.get_traffic_light_state()` to get the traffic light state.
        *   Update `self.current_traffic_light_state`.
        *   Set `self.is_waiting_at_red_light = True` if `self.current_traffic_light_state == carla.TrafficLightState.RED` AND vehicle `speed` is below a small threshold (e.g., `0.1` m/s). Otherwise, set to `False`.

    *   **B. Modify Idle Penalty Logic:**
        *   Locate the idle penalty section.
        *   If `self.is_waiting_at_red_light` is `True`, set `self.idle_penalty = 0.0` (or skip accumulation). Otherwise, apply existing logic.

    *   **C. Implement Positive Reward for Waiting at Red Light:**
        *   In reward calculation:
            *   `traffic_light_reward = 0.0`
            *   If `self.is_waiting_at_red_light` is `True`, then `traffic_light_reward = self.reward_wait_red_light`.
        *   Add `traffic_light_reward` to the total reward.

5.  **Ensure Imports:**
    *   `carla` is already imported, so `carla.TrafficLightState` is accessible.

**III. Visual Plan (Mermaid Diagram for `step` method logic flow)**

```mermaid
graph TD
    A[Start step(action)] --> B{Apply Control & Tick World};
    B --> C{Get Vehicle State (location, speed, etc.)};
    C --> D{Get Traffic Light State via vehicle.get_traffic_light_state()};
    D --> E[Store in self.current_traffic_light_state];
    E --> F{Is vehicle speed < threshold AND self.current_traffic_light_state == RED?};
    F -- Yes --> G[Set self.is_waiting_at_red_light = True];
    F -- No --> H[Set self.is_waiting_at_red_light = False];
    G --> I{Calculate Idle Penalty};
    H --> I;
    I -- self.is_waiting_at_red_light == True --> J[Set idle_penalty = 0.0];
    I -- self.is_waiting_at_red_light == False --> K[Calculate idle_penalty based on existing logic];
    J --> L{Calculate Rewards};
    K --> L;
    L -- self.is_waiting_at_red_light == True --> M[Add self.reward_wait_red_light to total_reward];
    L -- self.is_waiting_at_red_light == False --> N[No additional traffic light reward];
    M --> O[Update previous_location, previous_speed, etc.];
    N --> O;
    O --> P{Construct Observation (obs)};
    P --> Q[Return obs, reward, done, info];