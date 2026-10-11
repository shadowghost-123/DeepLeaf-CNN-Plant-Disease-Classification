import io
import json
import os
import threading
import warnings

import numpy as np
from PIL import Image

LEAF_CONFIDENCE_THRESHOLD = 0.40
HEATMAP_ACTIVATION_THRESHOLD = 0.5
SMALL_ATTENTION_MAX_PERCENT = 10.0
MODERATE_ATTENTION_MAX_PERCENT = 30.0

CLASS_GUIDANCE = {
    "Apple___Apple_scab": (
        "Remove and discard fallen leaves and prune scabbed shoots to reduce active spores. If lesions continue to spread, use an apple-scab-labelled fungicide such as captan or a locally approved alternative according to its label and local resistance guidance.",
        "Plant scab-resistant apple cultivars where suitable and prune annually to open the canopy. Shred or remove fallen leaves and avoid prolonged leaf wetness by watering at the soil line.",
    ),
    "Apple___Black_rot": (
        "Prune out cankers and remove mummified fruit and infected branches, cutting into healthy wood and disinfecting tools between cuts. Protect susceptible trees with a black-rot-labelled fungicide during locally recommended infection periods, following the product label.",
        "Remove dead wood, mummified apples, and nearby wild or abandoned apple debris before spring growth. Maintain an open canopy with balanced pruning and avoid wounds from machinery or careless pruning.",
    ),
    "Apple___Cedar_apple_rust": (
        "Remove visible orange-spotted leaves or fruit when practical, especially on young trees, and dispose of them away from the orchard. Where the disease is established, apply an apple-labelled rust fungicide at the locally recommended timing around spring infection, following the label.",
        "Choose cedar-apple-rust-resistant apple cultivars and remove nearby infected juniper galls when feasible and permitted. Space and prune trees for quick drying, and monitor new growth during wet spring weather.",
    ),
    "Apple___healthy": (
        "No treatment is needed for a healthy apple tree; continue routine observation rather than applying disease sprays without a diagnosis. Remove broken or dead branches with clean pruning tools if they appear.",
        "Water deeply at the root zone during dry periods and keep mulch away from the trunk. Prune in the dormant season to balance the canopy, thin fruit when needed, and collect fallen leaves and fruit.",
    ),
    "Blueberry___healthy": (
        "No treatment is needed for a healthy blueberry plant; avoid preventive pesticide applications without a confirmed pest or disease. Remove only dead or damaged canes with clean shears.",
        "Maintain acidic, well-drained soil with organic mulch and steady moisture, since blueberry roots dry quickly. Prune older canes to renew growth and protect blossoms from late frost where practical.",
    ),
    "Cherry_(including_sour)___Powdery_mildew": (
        "Prune and discard heavily mildewed shoots and remove infected leaves without composting them in a cool pile. If new growth remains affected, use a cherry-labelled powdery-mildew fungicide such as sulfur where permitted, applying strictly by its label and avoiding heat-sensitive conditions.",
        "Select powdery-mildew-tolerant cherry cultivars and prune to improve sunlight and air movement through the canopy. Avoid excess nitrogen that produces dense, soft shoots and monitor young leaves during humid weather.",
    ),
    "Cherry_(including_sour)___healthy": (
        "No treatment is needed for a healthy cherry tree; do not spray unless monitoring identifies a specific problem. Remove dead, crossing, or broken branches with clean tools at an appropriate pruning time.",
        "Water at the root zone during dry spells and keep the trunk flare clear of mulch. Prune to maintain a well-lit canopy, remove fallen fruit, and protect young trunks from mower and rodent injury.",
    ),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (
        "Scout lower leaves and, where lesions are expanding, use a corn-labelled foliar fungicide with activity against gray leaf spot, following local extension timing and label directions. Manage volunteer corn and infected residue after harvest rather than relying on sprays alone.",
        "Rotate corn with non-host crops and choose hybrids rated for gray leaf spot resistance. Bury or manage infected residue where appropriate, control volunteer corn, and avoid overly dense stands that keep leaves wet.",
    ),
    "Corn_(maize)___Common_rust_": (
        "Check pustules on several plants before treating; if rust threatens susceptible corn before silking, apply a corn-labelled triazole or strobilurin fungicide only when local thresholds and label directions support it. Preserve green leaf area and avoid unnecessary late sprays that cannot protect yield.",
        "Grow locally adapted common-rust-resistant hybrids and plant within the locally recommended window. Scout early because windborne spores can arrive from distant fields, and manage volunteer corn that can maintain infection between crops.",
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "For susceptible hybrids with spreading cigar-shaped lesions, apply a corn-labelled fungicide before disease reaches the ear leaf, guided by local thresholds and the product label. Prioritize protecting the upper canopy and do not use a foliar spray as a substitute for resistant seed.",
        "Choose hybrids with strong northern leaf blight ratings and rotate out of corn for at least one season where disease pressure is high. Manage infected residue and volunteer corn, and scout lower leaves before lesions move upward.",
    ),
    "Corn_(maize)___healthy": (
        "No treatment is needed for healthy corn; continue scouting rather than applying a fungicide without disease evidence. Correct nutrient or water stress only when field observations or soil tests identify a need.",
        "Use locally adapted seed and rotate corn with legumes or other non-host crops. Plant at recommended density, maintain balanced fertility based on soil tests, and manage volunteer corn and residue according to local disease pressure.",
    ),
    "Grape___Black_rot": (
        "Remove and destroy mummified berries, infected clusters, and diseased canes, including material left on the trellis or ground. If infection risk continues, apply a grape-labelled black-rot fungicide during the locally advised protection window and follow label restrictions around harvest.",
        "Remove overwintering mummies and prune vines to improve light and airflow around clusters. Train shoots vertically, control nearby wild grape vines where feasible, and use resistant cultivars for new plantings in high-pressure areas.",
    ),
    "Grape___Esca_(Black_Measles)": (
        "Mark affected vines and prune out dead arms during dry weather, cutting well below discolored wood and disinfecting tools between vines. Remove severely declining vines and their debris; there is no reliably curative foliar spray for established Esca infections.",
        "Protect large pruning wounds with an approved wound sealant where local guidance recommends it and avoid pruning immediately before wet weather. Use clean nursery stock, minimize trunk injuries, and remove dead vine wood that can harbor trunk-disease fungi.",
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (
        "Remove blighted leaves and fallen infected vine debris to reduce the source of spores. If lesions continue during wet weather, use a grape-labelled protectant fungicide effective against leaf spots, timed and applied according to local guidance and its label.",
        "Train and prune vines so foliage dries rapidly after rain and irrigation. Avoid overhead watering, remove infected leaves after harvest, and maintain balanced vine vigor rather than excessive nitrogen-driven canopy growth.",
    ),
    "Grape___healthy": (
        "No treatment is needed for healthy grapevines; avoid routine pesticide use without a monitored disease or pest risk. Remove dead or damaged canes with clean pruning tools during normal vine maintenance.",
        "Train shoots on a suitable trellis and thin excess growth to let sunlight and air reach clusters. Irrigate at the root zone, keep weeds from competing with young vines, and remove mummified berries after harvest.",
    ),
    "Orange___Haunglongbing_(Citrus_greening)": (
        "There is no curative treatment for huanglongbing; confirm suspected cases through local plant-health services and follow quarantine and tree-removal rules. Remove confirmed infected trees as directed and manage Asian citrus psyllids only with locally approved integrated pest-control measures.",
        "Plant certified HLB-free nursery trees and inspect new flush for Asian citrus psyllids and their nymphs. Coordinate monitoring and vector management with neighboring growers and local authorities, and do not move citrus plants or budwood from regulated areas.",
    ),
    "Peach___Bacterial_spot": (
        "Prune severely affected shoots and remove fallen diseased leaves, avoiding work on wet trees that can spread bacteria. Copper or other peach-labelled bactericides may suppress infection when locally advised, but use label rates and observe resistance and fruit-safety restrictions.",
        "Plant bacterial-spot-resistant peach cultivars suited to the region and use certified nursery stock. Keep foliage dry with drip irrigation, maintain open spacing for rapid drying, and avoid windbreak or pruning practices that injure leaves and fruit.",
    ),
    "Peach___healthy": (
        "No treatment is needed for a healthy peach tree; do not apply fungicides or bactericides without a diagnosed risk. Remove broken or dead wood with clean tools and keep routine pruning cuts small.",
        "Water deeply near the root zone during dry weather and maintain mulch without piling it against the trunk. Thin fruit to reduce branch stress, prune for an open vase shape, and clear fallen fruit to discourage pests.",
    ),
    "Pepper,_bell___Bacterial_spot": (
        "Remove badly spotted leaves and isolate affected transplants; do not handle plants while foliage is wet. A copper-based bactericide labelled for peppers may slow spread when applied early, but follow its label and local guidance because resistant bacterial strains are common.",
        "Start with certified disease-free seed and transplants, and rotate peppers away from tomato and other susceptible solanaceous crops for at least two seasons. Use drip irrigation, sanitize stakes and tools, and avoid working in wet foliage.",
    ),
    "Pepper,_bell___healthy": (
        "No treatment is needed for healthy bell peppers; inspect plants regularly and use pesticides only for an identified problem. Remove damaged leaves by hand and keep harvesting tools clean.",
        "Plant peppers in warm, well-drained soil with spacing that allows airflow and sunlight. Water consistently at the soil line, mulch to limit splash, and rotate away from other nightshades in the following seasons.",
    ),
    "Potato___Early_blight": (
        "Remove heavily blighted lower leaves and keep harvested tubers separate from visibly diseased vines. If disease is progressing, use a potato-labelled early-blight fungicide such as a protectant or locally registered mixture, rotating modes of action and following the label.",
        "Rotate potatoes with non-host crops for at least two to three years and destroy volunteer potatoes and cull piles. Use healthy seed tubers, maintain adequate nutrition without excess nitrogen, and irrigate at the soil line to reduce leaf wetness.",
    ),
    "Potato___Late_blight": (
        "Treat suspected late blight as urgent: isolate affected plants, avoid moving wet foliage, and destroy confirmed infected plants and tubers according to local guidance. Contact extension or plant-health authorities promptly and use only locally registered late-blight fungicides as part of a label-compliant program.",
        "Plant certified disease-free seed potatoes and resistant varieties, and never save tubers from infected crops. Destroy volunteer potatoes and cull piles, monitor regional late-blight alerts, and avoid overhead irrigation and prolonged leaf wetness.",
    ),
    "Potato___healthy": (
        "No treatment is needed for healthy potato plants; do not apply fungicides without a disease-risk reason. Hill soil around stems as needed to protect developing tubers and remove damaged foliage carefully.",
        "Use certified seed potatoes and rotate fields away from potatoes and tomatoes for several years. Maintain even soil moisture, hill adequately to prevent greening, and destroy volunteer plants and cull piles after harvest.",
    ),
    "Raspberry___healthy": (
        "No treatment is needed for healthy raspberry canes; continue inspection and avoid preventive pesticide applications without a target. Remove only dead floricanes after fruiting and keep pruning tools clean.",
        "Maintain a narrow, well-spaced row so leaves dry quickly and pick fruit frequently to limit rot and wildlife losses. Mulch shallow roots, water at the soil line, and remove old fruiting canes after harvest.",
    ),
    "Soybean___healthy": (
        "No treatment is needed for healthy soybean plants; avoid foliar fungicides unless scouting and local thresholds show a likely benefit. Correct nutrient issues only when a soil test or crop assessment supports the application.",
        "Plant locally adapted, disease-rated seed at the recommended date and seeding rate. Rotate with non-host crops, manage volunteer soybeans, and scout canopy and field edges regularly for early stress or disease symptoms.",
    ),
    "Squash___Powdery_mildew": (
        "Remove the most heavily mildewed leaves without stripping the plant and discard them away from the bed. For continuing spread, apply a squash-labelled powdery-mildew product such as sulfur or potassium bicarbonate where registered, following its label and avoiding heat-sensitive use conditions.",
        "Choose powdery-mildew-resistant squash varieties and space vines to improve airflow through the canopy. Avoid excess nitrogen, remove crop residue after harvest, and use drip irrigation so leaves are not kept wet overnight.",
    ),
    "Strawberry___Leaf_scorch": (
        "Remove scorched and dead leaves after harvest, taking care not to damage crowns, and dispose of infected debris. If the disease is confirmed and continues spreading, use only a strawberry-labelled fungicide recommended locally for leaf scorch, respecting the preharvest interval.",
        "Use certified disease-free strawberry plants and resistant cultivars when available, and renew beds before plants become crowded. Maintain row spacing and weed control for airflow, irrigate early at the soil line, and remove old leaves after harvest.",
    ),
    "Strawberry___healthy": (
        "No treatment is needed for healthy strawberry plants; avoid fungicide applications without a diagnosed disease or local risk advisory. Remove damaged leaves and ripe or overripe berries during routine picking.",
        "Plant certified stock in well-drained beds with rows spaced for airflow and sunlight. Renew older plantings, mulch berries to keep fruit off soil, and use drip irrigation to maintain even moisture without wetting foliage.",
    ),
    "Tomato___Bacterial_spot": (
        "Remove badly affected leaves and discard them; avoid handling or pruning plants while they are wet. Copper-based bactericides labelled for tomatoes may reduce new infections when applied early, but observe label directions and local advice because copper-resistant strains occur.",
        "Use certified disease-free seed or transplants and rotate tomatoes away from peppers and other solanaceous crops for at least two years. Stake plants for airflow, mulch to limit soil splash, and irrigate at the base rather than overhead.",
    ),
    "Tomato___Early_blight": (
        "Prune lower leaves with expanding target-like lesions and remove fallen infected debris from around the plants. If warm, wet conditions keep disease spreading, use a tomato-labelled early-blight fungicide such as chlorothalonil where locally registered, following the label and preharvest interval.",
        "Rotate tomatoes away from potatoes and other nightshades for at least two seasons and remove volunteer plants. Stake or cage plants, mulch before fruit sets, and water at the soil line to keep lower leaves from repeated splash.",
    ),
    "Tomato___Late_blight": (
        "Immediately isolate suspected plants and do not compost or transport infected foliage; confirm the diagnosis with local extension because late blight can spread rapidly. Remove and destroy confirmed infected plants and use only locally registered late-blight products as advised, observing all label restrictions.",
        "Plant healthy transplants and resistant varieties when available, and monitor regional late-blight alerts during cool, wet weather. Keep foliage dry with drip irrigation, space plants for airflow, and remove volunteer potatoes and tomatoes that can carry infection.",
    ),
    "Tomato___Leaf_Mold": (
        "Remove leaves with active olive-green mold on their undersides and discard them away from the greenhouse or garden. Improve ventilation immediately; if the disease persists, apply a tomato-labelled leaf-mold fungicide only where registered and according to its label.",
        "Choose leaf-mold-resistant tomato cultivars, especially for protected cultivation, and maintain lower humidity with ventilation and adequate plant spacing. Use drip irrigation, prune dense lower foliage, and sanitize greenhouse surfaces and crop debris between plantings.",
    ),
    "Tomato___Septoria_leaf_spot": (
        "Remove lower leaves bearing small gray-centered spots and collect fallen leaves without working through wet plants. When wet weather favors continued spread, use a tomato-labelled protectant fungicide such as chlorothalonil where registered, following the label and harvest interval.",
        "Rotate tomatoes away from other solanaceous crops for two to three years and remove volunteer tomato plants. Stake plants, mulch soil before rain, and irrigate at the base to prevent infected soil from splashing onto lower leaves.",
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Rinse leaf undersides with a firm stream of water and remove leaves with severe stippling and webbing. If mites persist, use a tomato-labelled insecticidal soap or horticultural oil with thorough underside coverage, following the label and avoiding application during heat or drought stress.",
        "Reduce plant water stress with consistent root-zone irrigation and avoid excessive broad-spectrum insecticides that kill mite predators. Inspect leaf undersides regularly, remove dusty weeds near crops, and conserve beneficial predatory mites where practical.",
    ),
    "Tomato___Target_Spot": (
        "Prune leaves with expanding target-like lesions and remove infected crop debris promptly. If disease is active in warm, humid conditions, apply a tomato-labelled fungicide effective against target spot as locally advised, rotating modes of action and observing the label.",
        "Use clean transplants and rotate tomatoes with non-host crops rather than peppers or potatoes. Space and stake plants to improve airflow, irrigate at the soil line, and remove volunteer tomatoes and residue after harvest.",
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "There is no curative treatment for tomato yellow leaf curl virus; remove and bag symptomatic plants promptly to reduce whitefly acquisition and spread. Manage whiteflies with an integrated, tomato-labelled approach recommended locally, protecting beneficial insects and following all product directions.",
        "Plant resistant tomato varieties and use insect-proof row covers or greenhouse screens while plants are young. Remove nearby infected tomatoes and host weeds, monitor whiteflies with traps, and avoid moving infested transplants between plantings.",
    ),
    "Tomato___Tomato_mosaic_virus": (
        "There is no chemical cure for tomato mosaic virus; remove and discard infected plants and do not save their seed. Wash hands and disinfect tools and stakes with a product effective against plant viruses before handling healthy tomato plants.",
        "Use certified virus-free seed and resistant tomato varieties, and avoid tobacco products or contaminated hands when pruning. Sanitize trays, tools, and greenhouse surfaces between crops, and remove volunteer tomatoes and solanaceous weeds.",
    ),
    "Tomato___healthy": (
        "No treatment is needed for healthy tomato plants; continue scouting and avoid routine sprays without a diagnosed issue. Remove broken or yellowing lower leaves with clean tools while keeping the plant canopy intact.",
        "Stake or cage tomatoes and space them for sunlight and airflow, then mulch to reduce soil splash. Water evenly at the root zone, rotate away from nightshades, and harvest ripe fruit regularly.",
    ),
}


def attention_area(percentage):
    if percentage < SMALL_ATTENTION_MAX_PERCENT:
        return "a small area"
    if percentage < MODERATE_ATTENTION_MAX_PERCENT:
        return "a moderate area"
    return "a large area"


def build_result_explanation(confidence, attention_percentage):
    area = attention_area(attention_percentage)
    return (
        f"The model predicted this class with {confidence:.1%} confidence and attended to {area} "
        f"of the leaf ({attention_percentage:.1f}% of Grad-CAM heatmap activations were above "
        f"{HEATMAP_ACTIVATION_THRESHOLD:.1f})."
    )


def _default_guidance():
    return (
        "Inspect the leaf carefully and compare symptoms to local extension guidance before choosing a treatment.",
        "Remove affected leaves and monitor plant health closely to reduce spread and protect nearby plants.",
    )


class Predictor:
    def __init__(self):
        self.model = None
        self.class_indices = {}
        self.index_to_class = {}
        self.model_path = None
        self.inference_lock = threading.Lock()

    @property
    def input_size(self):
        """Use the trained model's dimensions rather than assuming a fixed resize."""
        shape = getattr(self.model, "input_shape", None)
        if isinstance(shape, list):
            shape = shape[0]
        if shape and len(shape) >= 3 and shape[1] and shape[2]:
            return int(shape[1]), int(shape[2])
        return 128, 128

    def configure(self, model_path, indices_path):
        if self.model is not None and self.model_path == model_path:
            return
        self.model_path = model_path
        try:
            with open(indices_path, encoding="utf-8") as f:
                self.class_indices = json.load(f)
            self.index_to_class = {v: k for k, v in self.class_indices.items()}
            if set(self.class_indices) != set(CLASS_GUIDANCE):
                missing = sorted(set(self.class_indices) - set(CLASS_GUIDANCE))
                extra = sorted(set(CLASS_GUIDANCE) - set(self.class_indices))
                warnings.warn(
                    "Class guidance does not match the model classes; missing=%s, extra=%s" % (missing, extra),
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Handle both .keras directory format and legacy .h5 files
            if os.path.isdir(model_path):
                # Keras 3.x directory format (.keras)
                config_path = os.path.join(model_path, 'config.json')
                weights_path = os.path.join(model_path, 'model.weights.h5')

                if os.path.exists(config_path) and os.path.exists(weights_path):
                    with open(config_path, encoding="utf-8") as cf:
                        config_data = json.load(cf)

                    # Extract the nested config dict (contains 'name', 'layers', etc.)
                    model_config = config_data.get('config', config_data)

                    from tensorflow import keras
                    # Use Model.from_config() to reconstruct the Functional model
                    self.model = keras.Model.from_config(model_config)
                    # Load the trained weights
                    self.model.load_weights(weights_path)
                    # Compile for inference
                    self.model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                else:
                    raise ValueError(f"Missing config or weights files in {model_path}")
            else:
                # Legacy .h5 format
                from tensorflow.keras.models import load_model
                if os.path.exists(model_path):
                    self.model = load_model(model_path, compile=False)
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception:
            self.model = None
            raise

    @staticmethod
    def category(label):
        lower = label.lower()
        if "healthy" in lower:
            return "Healthy Leaf"
        if any(x in lower for x in ("virus", "mosaic", "curl")):
            return "Viral Disease"
        if "bacterial" in lower:
            return "Bacterial Disease"
        if any(x in lower for x in ("fungal", "scab", "rust", "blight", "mildew", "spot", "rot")):
            return "Fungal Disease"
        return "Leaf Condition"

    def preprocess(self, raw):
        input_height, input_width = self.input_size
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert("RGB").resize(
                (input_width, input_height),
                resample=Image.Resampling.BILINEAR,
            )
        return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

    def predict(self, raw, progress_callback=None):
        def progress(step, state):
            if progress_callback:
                progress_callback(step, state)

        if self.model is None:
            raise RuntimeError("The trained model is not available. Check that the supplied model file is present.")
        with self.inference_lock:
            progress(1, "in_progress")
            batch = self.preprocess(raw)
            from tensorflow.keras.applications.resnet50 import preprocess_input
            processed = preprocess_input(batch.copy())
            progress(1, "done")
            progress(2, "in_progress")
            probabilities = np.asarray(self.model.predict(processed, verbose=0))[0]
            max_confidence = float(np.max(probabilities))
            predicted_index = int(np.argmax(probabilities))
            predicted_name = self.index_to_class[predicted_index]
            confidence = float(probabilities[predicted_index])
            progress(2, "done")
            progress(3, "in_progress")
            heatmap = self.grad_cam(processed, predicted_index)
            attention_percentage = float(
                np.count_nonzero(heatmap > HEATMAP_ACTIVATION_THRESHOLD) * 100.0 / heatmap.size
            )
            severity = "Severe" if float((heatmap > 0.6).mean()) > .25 else ("Moderate" if float((heatmap > .4).mean()) > .10 else "Mild")
            progress(3, "done")
            category = self.category(predicted_name)
            treatment, prevention = self.guidance(predicted_name)
        return {
            "label": predicted_name.replace("___", " — ").replace("_", " "),
            "raw_label": predicted_name,
            "category": category,
            "confidence": confidence,
            "max_confidence": max_confidence,
            "severity": severity,
            "heatmap": heatmap,
            "attention_percentage": attention_percentage,
            "explanation": build_result_explanation(confidence, attention_percentage),
            "treatment": treatment,
            "prevention": prevention,
        }

    @staticmethod
    def guidance(label):
        guidance = CLASS_GUIDANCE.get(label)
        if guidance is not None:
            return guidance
        return _default_guidance()

    def grad_cam(self, batch, class_index):
        import tensorflow as tf

        conv = None
        for layer in reversed(self.model.layers):
            output_shape = getattr(layer, "output_shape", None)
            if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 4:
                conv = layer
                break
        if conv is None:
            raise TypeError("Grad-CAM requires a convolutional feature map to compute attention.")

        grad_model = tf.keras.Model(self.model.inputs, [conv.output, self.model.output])
        with tf.GradientTape() as tape:
            activations, outputs = grad_model(batch, training=False)
            score = outputs[:, class_index]
        grads = tape.gradient(score, activations)
        if grads is None:
            raise RuntimeError("Grad-CAM could not compute gradients for the selected class.")
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(activations * weights[:, None, None, :], axis=-1)[0]
        cam = tf.maximum(cam, 0)
        max_value = tf.reduce_max(cam)
        if tf.equal(max_value, 0):
            return tf.zeros_like(cam).numpy()
        return (cam / (max_value + 1e-8)).numpy()


predictor = Predictor()
