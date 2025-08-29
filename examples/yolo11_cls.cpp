// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolo11-cls torchscript
//      yolo export model=yolo11n-cls.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolo11n-cls.torchscript
// 4. now you get ncnn model files
//      yolo11n_cls.ncnn.param
//      yolo11n_cls.ncnn.bin

#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

struct Object
{
    int label;
    float prob;
};

static void get_topk(const ncnn::Mat& cls_scores, int topk, std::vector<Object>& objects)
{
    // partial sort topk with index
    int size = cls_scores.w;
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    objects.resize(topk);
    for (int i = 0; i < topk; i++)
    {
        objects[i].label = vec[i].second;
        objects[i].prob = vec[i].first;
    }
}

static int detect_yolo11_cls(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolo11;

    yolo11.opt.use_vulkan_compute = true;
    // yolo11.opt.use_bf16_storage = true;

    // https://github.com/nihui/ncnn-android-yolo11/tree/master/app/src/main/assets
    yolo11.load_param("yolo11n_cls.ncnn.param");
    yolo11.load_model("yolo11n_cls.ncnn.bin");
    // yolo11.load_param("yolo11s_cls.ncnn.param");
    // yolo11.load_model("yolo11s_cls.ncnn.bin");
    // yolo11.load_param("yolo11m_cls.ncnn.param");
    // yolo11.load_model("yolo11m_cls.ncnn.bin");

    const int target_size = 224;
    const int topk = 5;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // letterbox pad to target_size rectangle
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo11.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    // return top-5
    get_topk(out, topk, objects);

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock",
        "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "robin", "bulbul",
        "jay", "magpie", "chickadee", "water ouzel", "kite", "bald eagle", "vulture", "great grey owl",
        "European fire salamander", "common newt", "eft", "spotted salamander", "axolotl", "bullfrog", "tree frog",
        "tailed frog", "loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle", "banded gecko",
        "common iguana", "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
        "Gila monster", "green lizard", "African chameleon", "Komodo dragon", "African crocodile",
        "American alligator", "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake",
        "king snake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "rock python",
        "Indian cobra", "green mamba", "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
        "harvestman", "scorpion", "black and gold garden spider", "barn spider", "garden spider", "black widow",
        "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse",
        "prairie chicken", "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "drake",
        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala",
        "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug",
        "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "king crab",
        "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork",
        "spoonbill", "flamingo", "little blue heron", "American egret", "bittern", "crane (bird)", "limpkin",
        "European gallinule", "American coot", "bustard", "ruddy turnstone", "red-backed sandpiper", "redshank",
        "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale",
        "dugong", "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu",
        "Blenheim spaniel", "papillon", "toy terrier", "Rhodesian ridgeback", "Afghan hound", "basset", "beagle",
        "bloodhound", "bluetick", "black-and-tan coonhound", "Walker hound", "English foxhound", "redbone",
        "borzoi", "Irish wolfhound", "Italian greyhound", "whippet", "Ibizan hound", "Norwegian elkhound",
        "otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
        "American Staffordshire terrier", "Bedlington terrier", "Border terrier", "Kerry blue terrier",
        "Irish terrier", "Norfolk terrier", "Norwich terrier", "Yorkshire terrier", "wire-haired fox terrier",
        "Lakeland terrier", "Sealyham terrier", "Airedale", "cairn", "Australian terrier", "Dandie Dinmont",
        "Boston bull", "miniature schnauzer", "giant schnauzer", "standard schnauzer", "Scotch terrier",
        "Tibetan terrier", "silky terrier", "soft-coated wheaten terrier", "West Highland white terrier",
        "Lhasa", "flat-coated retriever", "curly-coated retriever", "golden retriever", "Labrador retriever",
        "Chesapeake Bay retriever", "German short-haired pointer", "vizsla", "English setter", "Irish setter",
        "Gordon setter", "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel",
        "cocker spaniel", "Sussex spaniel", "Irish water spaniel", "kuvasz", "schipperke", "groenendael",
        "malinois", "briard", "kelpie", "komondor", "Old English sheepdog", "Shetland sheepdog", "collie",
        "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd", "Doberman",
        "miniature pinscher", "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller", "EntleBucher",
        "boxer", "bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane", "Saint Bernard",
        "Eskimo dog", "malamute", "Siberian husky", "dalmatian", "affenpinscher", "basenji", "pug", "Leonberg",
        "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond", "Brabancon griffon",
        "Pembroke", "Cardigan", "toy poodle", "miniature poodle", "standard poodle", "Mexican hairless",
        "timber wolf", "white wolf", "red wolf", "coyote", "dingo", "dhole", "African hunting dog", "hyena",
        "red fox", "kit fox", "Arctic fox", "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat",
        "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah",
        "brown bear", "American black bear", "ice bear", "sloth bear", "mongoose", "meerkat", "tiger beetle",
        "ladybug", "ground beetle", "long-horned beetle", "leaf beetle", "dung beetle", "rhinoceros beetle",
        "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking stick", "cockroach", "mantis",
        "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "admiral", "ringlet", "monarch",
        "cabbage butterfly", "sulphur butterfly", "lycaenid", "starfish", "sea urchin", "sea cucumber",
        "wood rabbit", "hare", "Angora", "hamster", "porcupine", "fox squirrel", "marmot", "beaver",
        "guinea pig", "sorrel", "zebra", "hog", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo",
        "bison", "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle", "Arabian camel", "llama",
        "weasel", "mink", "polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo",
        "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas",
        "baboon", "macaque", "langur", "colobus", "proboscis monkey", "marmoset", "capuchin", "howler monkey",
        "titi", "spider monkey", "squirrel monkey", "Madagascar cat", "indri", "Indian elephant",
        "African elephant", "lesser panda", "giant panda", "barracouta", "eel", "coho", "rock beauty",
        "anemone fish", "sturgeon", "gar", "lionfish", "puffer", "abacus", "abaya", "academic gown",
        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
        "amphibian", "analog clock", "apiary", "apron", "ashcan", "assault rifle", "backpack", "bakery",
        "balance beam", "balloon", "ballpoint", "Band Aid", "banjo", "bannister", "barbell", "barber chair",
        "barbershop", "barn", "barometer", "barrel", "barrow", "baseball", "basketball", "bassinet", "bassoon",
        "bathing cap", "bath towel", "bathtub", "beach wagon", "beacon", "beaker", "bearskin", "beer bottle",
        "beer glass", "bell cote", "bib", "bicycle-built-for-two", "bikini", "binder", "binoculars",
        "birdhouse", "boathouse", "bobsled", "bolo tie", "bonnet", "bookcase", "bookshop", "bottlecap", "bow",
        "bow tie", "brass", "brassiere", "breakwater", "breastplate", "broom", "bucket", "buckle",
        "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron", "candle", "cannon", "canoe",
        "can opener", "cardigan", "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
        "cash machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello",
        "cellular telephone", "chain", "chainlink fence", "chain mail", "chain saw", "chest", "chiffonier",
        "chime", "china cabinet", "Christmas stocking", "church", "cinema", "cleaver", "cliff dwelling",
        "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock",
        "computer keyboard", "confectionery", "container ship", "convertible", "corkscrew", "cornet",
        "cowboy boot", "cowboy hat", "cradle", "crane (machine)", "crash helmet", "crate", "crib",
        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "dial telephone",
        "diaper", "digital clock", "digital watch", "dining table", "dishrag", "dishwasher", "disk brake",
        "dock", "dogsled", "dome", "doormat", "drilling platform", "drum", "drumstick", "dumbbell",
        "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center",
        "envelope", "espresso maker", "face powder", "feather boa", "file", "fireboat", "fire engine",
        "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain",
        "fountain pen", "four-poster", "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
        "gasmask", "gas pump", "goblet", "go-kart", "golf ball", "golfcart", "gondola", "gong", "gown",
        "grand piano", "greenhouse", "grille", "grocery store", "guillotine", "hair slide", "hair spray",
        "half track", "hammer", "hamper", "hand blower", "hand-held computer", "handkerchief", "hard disc",
        "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb", "hook",
        "hoopskirt", "horizontal bar", "horse cart", "hourglass", "iPod", "iron", "jack-o'-lantern", "jean",
        "jeep", "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono", "knee pad", "knot", "lab coat",
        "ladle", "lampshade", "laptop", "lawn mower", "lens cap", "letter opener", "library", "lifeboat",
        "lighter", "limousine", "liner", "lipstick", "Loafer", "lotion", "loudspeaker", "loupe", "lumbermill",
        "magnetic compass", "mailbag", "mailbox", "maillot (tights)", "maillot (tank suit)", "manhole cover",
        "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine chest",
        "megalith", "microphone", "microwave", "military uniform", "milk can", "minibus", "miniskirt",
        "minivan", "missile", "mitten", "mixing bowl", "mobile home", "Model T", "modem", "monastery",
        "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito net", "motor scooter", "mountain bike",
        "mountain tent", "mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace",
        "nipple", "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope",
        "overskirt", "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel", "padlock", "paintbrush",
        "pajama", "palace", "panpipe", "paper towel", "parachute", "parallel bars", "park bench",
        "parking meter", "passenger car", "patio", "pay-phone", "pedestal", "pencil box", "pencil sharpener",
        "perfume", "Petri dish", "photocopier", "pick", "pickelhaube", "picket fence", "pickup", "pier",
        "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate", "pitcher", "plane",
        "planetarium", "plastic bag", "plate rack", "plow", "plunger", "Polaroid camera", "pole",
        "police van", "poncho", "pool table", "pop bottle", "pot", "potter's wheel", "power drill",
        "prayer rug", "printer", "prison", "projectile", "projector", "puck", "punching bag", "purse",
        "quill", "quilt", "racer", "racket", "radiator", "radio", "radio telescope", "rain barrel",
        "recreational vehicle", "reel", "reflex camera", "refrigerator", "remote control", "restaurant",
        "revolver", "rifle", "rocking chair", "rotisserie", "rubber eraser", "rugby ball", "rule",
        "running shoe", "safe", "safety pin", "saltshaker", "sandal", "sarong", "sax", "scabbard", "scale",
        "school bus", "schooner", "scoreboard", "screen", "screw", "screwdriver", "seat belt", "sewing machine",
        "shield", "shoe shop", "shoji", "shopping basket", "shopping cart", "shovel", "shower cap",
        "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot", "snorkel",
        "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero",
        "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "speedboat", "spider web",
        "spindle", "sports car", "spotlight", "stage", "steam locomotive", "steel arch bridge", "steel drum",
        "stethoscope", "stole", "stone wall", "stopwatch", "stove", "strainer", "streetcar", "stretcher",
        "studio couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen",
        "suspension bridge", "swab", "sweatshirt", "swimming trunks", "swing", "switch", "syringe",
        "table lamp", "tank", "tape player", "teapot", "teddy", "television", "tennis ball", "thatch",
        "theater curtain", "thimble", "thresher", "throne", "tile roof", "toaster", "tobacco shop",
        "toilet seat", "torch", "totem pole", "tow truck", "toyshop", "tractor", "trailer truck", "tray",
        "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "tub",
        "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright", "vacuum", "vase", "vault",
        "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock",
        "wallet", "wardrobe", "warplane", "washbasin", "washer", "water bottle", "water jug", "water tower",
        "whiskey jug", "whistle", "wig", "window screen", "window shade", "Windsor tie", "wine bottle", "wing",
        "wok", "wooden spoon", "wool", "worm fence", "wreck", "yawl", "yurt", "web site", "comic book",
        "crossword puzzle", "street sign", "traffic light", "book jacket", "menu", "plate", "guacamole",
        "consomme", "hot pot", "trifle", "ice cream", "ice lolly", "French loaf", "bagel", "pretzel",
        "cheeseburger", "hotdog", "mashed potato", "head cabbage", "broccoli", "cauliflower", "zucchini",
        "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper",
        "cardoon", "mushroom", "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple", "banana",
        "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate sauce", "dough",
        "meat loaf", "pizza", "potpie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble",
        "cliff", "coral reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano",
        "ballplayer", "groom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
        "hip", "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods",
        "bolete", "ear", "toilet tissue"
    };

    cv::Mat image = bgr.clone();

    int y_offset = 0;
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f\n", obj.label, obj.prob);

        char text[256];
        sprintf(text, "%4.1f%% %s", obj.prob * 100, class_names[obj.label]);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = 0;
        int y = y_offset;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        y_offset += label_size.height;
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolo11_cls(m, objects);

    draw_objects(m, objects);

    return 0;
}
