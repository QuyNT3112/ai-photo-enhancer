"""
AI Franchise & Entity Knowledge Engine for Clarity AI Studio
Recognizes pop culture universes, anime/manga franchises, superheroes,
and extracts key characters, lore, and visual search query expansions.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple


def normalize_vietnamese_text(text: str) -> str:
    """Normalizes text for fuzzy match (lowercase, stripped, accents removed)."""
    text = text.lower().strip()
    # Normalize unicode
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = text.replace('đ', 'd').replace('Đ', 'd')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


FRANCHISE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "dragon_ball": {
        "id": "dragon_ball",
        "title": "Dragon Ball (Bảy Viên Ngọc Rồng)",
        "original_title": "ドラゴンボール",
        "creator": "Akira Toriyama (Nhật Bản)",
        "category": "Anime & Manga Huyền Thoại · Shonen Jump",
        "icon": "🐉",
        "badge": "Franchise Kinh Điển",
        "color_theme": "orange",
        "description": "Siêu phẩm manga & anime vĩ đại nhất lịch sử của cố tác giả Akira Toriyama. Câu chuyện kể về hành trình phiêu lưu của Son Goku cùng các chiến binh Z tìm kiếm 7 viên ngọc rồng, luyện tập võ thuật và bảo vệ vũ trụ.",
        "search_enrichment": "Dragon Ball Z Super anime manga official art Akira Toriyama characters",
        "keywords": [
            "dragon ball", "dragonball", "bay vien ngoc rong", "7 vien ngoc rong", "ngoc rong",
            "dbz", "dbs", "dragon ball z", "dragon ball super", "dragon ball gt",
            "songoku", "son goku", "goku", "vegeta", "cadic", "gohan", "trunks",
            "piccolo", "pocolo", "shenron", "rong than", "frieza", "fide", "cell",
            "xên bọ hung", "majin buu", "ma bu", "kamehameha", "saiyan", "super saiyan",
            "ultra instinct", "beerus", "whis", "krilin", "krillin", "yamcha", "tien shinhan"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Dragon Ball",
                "vietnamese_name": "Tất Cả Nhân Vật & Cảnh Phim",
                "icon": "🌟",
                "badge": "Toàn Bộ Universe",
                "search_query": "Dragon Ball Z anime characters wallpaper official art",
                "keywords": ["all", "tat ca", "dragon ball"]
            },
            {
                "id": "goku",
                "name": "Son Goku (Kakarot)",
                "vietnamese_name": "Son Goku (Khỉ Con)",
                "icon": "⚡",
                "badge": "Nhân vật chính · Siêu Saiyan",
                "search_query": "Dragon Ball Son Goku anime official art wallpaper",
                "keywords": ["goku", "son goku", "songoku", "kakarot", "khi con"]
            },
            {
                "id": "vegeta",
                "name": "Vegeta (Cadic)",
                "vietnamese_name": "Hoàng Tử Vegeta (Cadic)",
                "icon": "🔥",
                "badge": "Hoàng tử Saiyan · Kiêu hãnh",
                "search_query": "Dragon Ball Vegeta anime official art wallpaper",
                "keywords": ["vegeta", "cadic", "hoang tu saiyan"]
            },
            {
                "id": "gohan",
                "name": "Son Gohan",
                "vietnamese_name": "Son Gohan (Beast)",
                "icon": "💥",
                "badge": "Chiến binh Saiyan lai · Gohan Beast",
                "search_query": "Dragon Ball Son Gohan anime official art",
                "keywords": ["gohan", "son gohan", "gohan beast"]
            },
            {
                "id": "trunks",
                "name": "Trunks (Tương Lai)",
                "vietnamese_name": "Trunks Tương Lai (Kiếm Sĩ)",
                "icon": "⚔️",
                "badge": "Kiếm sĩ Saiyan tương lai",
                "search_query": "Dragon Ball Future Trunks anime sword official art",
                "keywords": ["trunks", "future trunks", "trunks tuong lai"]
            },
            {
                "id": "piccolo",
                "name": "Piccolo (Pô-cô-lô)",
                "vietnamese_name": "Đại Ma Vương Piccolo Namek",
                "icon": "🟢",
                "badge": "Đại ma vương Namek · Người thầy",
                "search_query": "Dragon Ball Piccolo Namek anime official art",
                "keywords": ["piccolo", "pocolo", "dai ma vuong piccolo", "namek"]
            },
            {
                "id": "shenron",
                "name": "Rồng Thần Shenron",
                "vietnamese_name": "Rồng Thần Ban Điều Ước (Shenron)",
                "icon": "🐉",
                "badge": "Rồng Thần Thượng Cổ",
                "search_query": "Dragon Ball Shenron Eternal Dragon anime art wallpaper",
                "keywords": ["shenron", "rong than", "eternal dragon", "than long"]
            },
            {
                "id": "dragon_balls",
                "name": "7 Viên Ngọc Rồng",
                "vietnamese_name": "7 Viên Ngọc Rồng Pha Lê",
                "icon": "🔮",
                "badge": "Bảo vật ước nguyện tối thượng",
                "search_query": "seven Dragon Balls crystal glowing stars anime",
                "keywords": ["ngoc rong", "dragon balls", "7 vien ngoc rong", "crystal ball"]
            },
            {
                "id": "frieza",
                "name": "Frieza (Fide)",
                "vietnamese_name": "Đế Vương Vũ Trụ Fide (Frieza)",
                "icon": "👾",
                "badge": "Đế vương vũ trụ · Golden Frieza",
                "search_query": "Dragon Ball Frieza Golden Frieza anime official art",
                "keywords": ["frieza", "fide", "golden frieza"]
            },
            {
                "id": "cell",
                "name": "Cell (Xên Bọ Hung)",
                "vietnamese_name": "Xên Bọ Hung Hoàn Hảo (Perfect Cell)",
                "icon": "🦠",
                "badge": "Sinh vật nhân tạo hoàn hảo",
                "search_query": "Dragon Ball Perfect Cell anime official art",
                "keywords": ["cell", "perfect cell", "xen bo hung"]
            },
            {
                "id": "majin_buu",
                "name": "Majin Buu (Ma Bư)",
                "vietnamese_name": "Ma Bư (Majin Buu / Kid Buu)",
                "icon": "🍬",
                "badge": "Ma nhân hỗn mang bí ẩn",
                "search_query": "Dragon Ball Majin Buu Kid Buu anime official art",
                "keywords": ["buu", "majin buu", "ma bu", "kid buu"]
            },
            {
                "id": "super_saiyan",
                "name": "Super Saiyan (SSJ & Ultra Instinct)",
                "vietnamese_name": "Trạng Thái Siêu Saiyan & Bản Năng Vô Cực",
                "icon": "✨",
                "badge": "Biến thân hào quang tối thượng",
                "search_query": "Dragon Ball Super Saiyan Ultra Instinct Blue glowing aura anime",
                "keywords": ["super saiyan", "ssj", "ultra instinct", "ban nang vo cuc", "ssj blue"]
            },
            {
                "id": "beerus",
                "name": "Thần Hủy Diệt Beerus & Whis",
                "vietnamese_name": "Thần Hủy Diệt Beerus & Thiên Sứ Whis",
                "icon": "🐱",
                "badge": "Thần Hủy Diệt & Thiên sứ vũ trụ 7",
                "search_query": "Dragon Ball Lord Beerus Whis God of Destruction anime",
                "keywords": ["beerus", "whis", "than huy diet"]
            }
        ]
    },

    "one_piece": {
        "id": "one_piece",
        "title": "One Piece (Đảo Hải Tặc / Vua Hải Tặc)",
        "original_title": "ワンピース",
        "creator": "Eiichiro Oda (Nhật Bản)",
        "category": "Anime & Manga Hải Tặc Kinh Điển",
        "icon": "🏴‍☠️",
        "badge": "Kỷ lục Guiness Thế Giới",
        "color_theme": "red",
        "description": "Bộ truyện tranh bán chạy nhất mọi thời đại của Eiichiro Oda kể về Monkey D. Luffy và băng Mũ Rơm giương buồm chinh phục Đại Hải Trình để tìm kiếm kho báu One Piece.",
        "search_enrichment": "One Piece anime manga Eiichiro Oda Straw Hat Pirates characters wallpaper",
        "keywords": [
            "one piece", "onepiece", "dao hai tac", "vua hai tac", "hai tac mu rom",
            "luffy", "monkey d luffy", "zoro", "roronoa zoro", "sanji", "nami",
            "robin", "nico robin", "chopper", "tony tony chopper", "usopp", "brook",
            "franky", "jinbe", "ace", "portgas d ace", "shanks", "shanks toc do",
            "gear 5", "nika", "whitebeard", "kaido", "trafalgar law"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả One Piece",
                "vietnamese_name": "Băng Mũ Rơm & Thế Giới Hải Tặc",
                "icon": "🏴‍☠️",
                "badge": "Toàn Bộ Universe",
                "search_query": "One Piece Straw Hat Pirates anime wallpaper",
                "keywords": ["all", "tat ca", "one piece"]
            },
            {
                "id": "luffy",
                "name": "Monkey D. Luffy (Gear 5 Nika)",
                "vietnamese_name": "Thuyền Trưởng Mũ Rơm Luffy",
                "icon": "👒",
                "badge": "Vua Hải Tặc tương lai · Thần Mặt Trời Nika",
                "search_query": "One Piece Monkey D Luffy Gear 5 anime wallpaper",
                "keywords": ["luffy", "monkey d luffy", "mu rom", "gear 5", "nika"]
            },
            {
                "id": "zoro",
                "name": "Roronoa Zoro (Tam Kiếm Phái)",
                "vietnamese_name": "Kiếm Sĩ Roronoa Zoro",
                "icon": "⚔️",
                "badge": "Đệ nhất kiếm sĩ tương lai · Tam Kiếm Phái",
                "search_query": "One Piece Roronoa Zoro three swords anime wallpaper",
                "keywords": ["zoro", "roronoa zoro", "tam kiem phai"]
            },
            {
                "id": "sanji",
                "name": "Vinsmoke Sanji (Chân Đen)",
                "vietnamese_name": "Đầu Bếp Sanji Chân Đen",
                "icon": "🦵",
                "badge": "Đầu bếp tài hoa · Hắc cước",
                "search_query": "One Piece Vinsmoke Sanji Ifrit Jambe anime",
                "keywords": ["sanji", "vinsmoke sanji", "chan den"]
            },
            {
                "id": "nami",
                "name": "Nami (Miêu Tặc)",
                "vietnamese_name": "Hoa Tiêu Nami",
                "icon": "🍊",
                "badge": "Hoa tiêu thiên tài · Zeus",
                "search_query": "One Piece Nami anime official art",
                "keywords": ["nami", "mieu tac"]
            },
            {
                "id": "robin",
                "name": "Nico Robin (Đứa Con Của Quỷ)",
                "vietnamese_name": "Nhà Khảo Cổ Nico Robin",
                "icon": "📖",
                "badge": "Nhà khảo cổ học · Giải mã Poneglyph",
                "search_query": "One Piece Nico Robin anime official art",
                "keywords": ["robin", "nico robin"]
            },
            {
                "id": "chopper",
                "name": "Tony Tony Chopper",
                "vietnamese_name": "Bác Sĩ Tuần Lộc Chopper",
                "icon": "🦌",
                "badge": "Bác sĩ đáng yêu của băng",
                "search_query": "One Piece Tony Tony Chopper cute anime",
                "keywords": ["chopper", "tony tony chopper"]
            },
            {
                "id": "ace",
                "name": "Portgas D. Ace (Hỏa Quyền)",
                "vietnamese_name": "Hỏa Quyền Ace",
                "icon": "🔥",
                "badge": "Đội trưởng đội 2 Băng Râu Trắng",
                "search_query": "One Piece Portgas D Ace fire fist anime",
                "keywords": ["ace", "portgas d ace", "hoa quyen ace"]
            },
            {
                "id": "shanks",
                "name": "Tứ Hoàng Shanks Tóc Đỏ",
                "vietnamese_name": "Tứ Hoàng Shanks",
                "icon": "🗡️",
                "badge": "Tứ hoàng biển cả · Haki Bá Vương",
                "search_query": "One Piece Red-Haired Shanks Yonko anime",
                "keywords": ["shanks", "shanks toc do", "tu hoang"]
            }
        ]
    },

    "naruto": {
        "id": "naruto",
        "title": "Naruto & Naruto Shippuden",
        "original_title": "ナルト",
        "creator": "Masashi Kishimoto (Nhật Bản)",
        "category": "Anime & Manga Ninja Huyền Thoại",
        "icon": "🍥",
        "badge": "Biểu Tượng Shonen",
        "color_theme": "yellow",
        "description": "Câu chuyện giàu cảm xúc về cậu bé mồ côi Naruto Uzumaki mang trong mình Cửu Vĩ Hồ Ly, không ngừng vượt qua định kiến để trở thành Hokage vĩ đại làng Lá.",
        "search_enrichment": "Naruto Shippuden anime manga Masashi Kishimoto ninja characters wallpaper",
        "keywords": [
            "naruto", "naruto shippuden", "hokage", "lang la", "ninja", "boruto",
            "uzumaki naruto", "sasuke", "uchiha sasuke", "sakura", "kakashi",
            "itachi", "uchiha itachi", "jiraiya", "tsunade", "orochimaru",
            "hinata", "madara", "uchiha madara", "gaara", "minato", "cuu vi", "kurama"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Naruto Universe",
                "vietnamese_name": "Làng Lá & Thế Giới Shinobi",
                "icon": "🍥",
                "badge": "Toàn Bộ Universe",
                "search_query": "Naruto Shippuden ninja characters wallpaper anime",
                "keywords": ["all", "tat ca", "naruto"]
            },
            {
                "id": "naruto",
                "name": "Uzumaki Naruto (Hokage Đệ Thất)",
                "vietnamese_name": "Uzumaki Naruto (Cửu Vĩ / Rasengan)",
                "icon": "🍥",
                "badge": "Hokage Đệ Thất làng Lá",
                "search_query": "Naruto Uzumaki Nine Tails Sage Mode Kurama anime",
                "keywords": ["naruto", "uzumaki naruto", "cuu vi", "rasengan"]
            },
            {
                "id": "sasuke",
                "name": "Uchiha Sasuke (Rinnegan & Sharingan)",
                "vietnamese_name": "Uchiha Sasuke",
                "icon": "⚡",
                "badge": "Hậu duệ tộc Uchiha · Chidori",
                "search_query": "Sasuke Uchiha Rinnegan Sharingan Chidori anime",
                "keywords": ["sasuke", "uchiha sasuke", "chidori", "rinnegan"]
            },
            {
                "id": "kakashi",
                "name": "Hatake Kakashi (Sharingan)",
                "vietnamese_name": "Thầy Kakashi Ninja Sao Chép",
                "icon": "⚡",
                "badge": "Ninja sao chép làng Lá · Hokage Đệ Lục",
                "search_query": "Kakashi Hatake Sharingan Chidori anime wallpaper",
                "keywords": ["kakashi", "hatake kakashi"]
            },
            {
                "id": "itachi",
                "name": "Uchiha Itachi (Susanoo & Mangekyo)",
                "vietnamese_name": "Uchiha Itachi (Thiên Tài Bí Ẩn)",
                "icon": "👁️",
                "badge": "Người anh hùng bóng tối tộc Uchiha",
                "search_query": "Itachi Uchiha Akatsuki Susanoo crows anime wallpaper",
                "keywords": ["itachi", "uchiha itachi", "akatsuki"]
            },
            {
                "id": "jiraiya",
                "name": "Jiraiya (Tiên Nhân Cóc)",
                "vietnamese_name": "Hiền Nhân Jiraiya (San-nin)",
                "icon": "🐸",
                "badge": "Một trong Tam Ninja Huyền Thoại",
                "search_query": "Jiraiya Toad Sage Naruto anime official art",
                "keywords": ["jiraiya", "tien nhan coc"]
            },
            {
                "id": "minato",
                "name": "Namikaze Minato (Tia Chớp Vàng)",
                "vietnamese_name": "Hokage Đệ Tứ Minato",
                "icon": "⚡",
                "badge": "Tia Chớp Vàng làng Lá · Cha của Naruto",
                "search_query": "Minato Namikaze Yellow Flash Fourth Hokage anime",
                "keywords": ["minato", "namikaze minato", "tia chop vang"]
            }
        ]
    },

    "doraemon": {
        "id": "doraemon",
        "title": "Doraemon (Chú Mèo Máy Đến Từ Tương Lai)",
        "original_title": "ドラえもん",
        "creator": "Fujiko F. Fujio (Nhật Bản)",
        "category": "Tuổi Thơ Huyền Thoại · Thiếu Nhi & Gia Đình",
        "icon": "🔔",
        "badge": "Biểu Tượng Văn Hóa Nhật Bản",
        "color_theme": "blue",
        "description": "Tác phẩm gắn liền với tuổi thơ nhiều thế hệ về chú mèo máy thông minh Doraemon cùng chiếc túi thần kỳ và người bạn hậu đậu Nobita.",
        "search_enrichment": "Doraemon Fujiko F Fujio anime movie Nobita Shizuka wallpaper",
        "keywords": [
            "doraemon", "doremon", "doremon", "nobita", "xuka", "shizuka",
            "xeko", "suneo", "chaien", "gian", "dorami", "chu meo may",
            "tui than ky", "chong chong tre", "canh cua than ky"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Doraemon",
                "vietnamese_name": "Nhóm Bạn Doraemon & Nobita",
                "icon": "🔔",
                "badge": "Toàn Bộ Universe",
                "search_query": "Doraemon Nobita Shizuka anime wallpaper",
                "keywords": ["all", "tat ca", "doraemon"]
            },
            {
                "id": "doraemon",
                "name": "Chú Mèo Máy Doraemon",
                "vietnamese_name": "Mèo Máy Doraemon",
                "icon": "🔔",
                "badge": "Mèo máy thế kỷ 22 · Bánh rán Dorayaki",
                "search_query": "Doraemon robotic cat anime official art",
                "keywords": ["doraemon", "meo may", "dorayaki"]
            },
            {
                "id": "nobita",
                "name": "Nobi Nobita",
                "vietnamese_name": "Cậu Bé Nobi Nobita",
                "icon": "👓",
                "badge": "Cậu bạn vụng về nhưng nhân hậu",
                "search_query": "Nobi Nobita Doraemon anime art",
                "keywords": ["nobita", "nobi nobita"]
            },
            {
                "id": "shizuka",
                "name": "Minamoto Shizuka (Xuka)",
                "vietnamese_name": "Cô Bạn Xuka (Shizuka)",
                "icon": "🌸",
                "badge": "Cô bạn xinh xắn, ngoan ngoãn",
                "search_query": "Shizuka Minamoto Doraemon anime",
                "keywords": ["shizuka", "xuka"]
            },
            {
                "id": "suneo",
                "name": "Honekawa Suneo (Xêkô)",
                "vietnamese_name": "Suneo Mỏ Nhọn (Xêkô)",
                "icon": "🦊",
                "badge": "Cậu ấm nhà giàu mê đồ chơi công nghệ",
                "search_query": "Suneo Honekawa Doraemon anime",
                "keywords": ["suneo", "xeko"]
            },
            {
                "id": "gian",
                "name": "Goda Takeshi (Chaien / Gian)",
                "vietnamese_name": "Chaien (Takeshi Gian)",
                "icon": "🎤",
                "badge": "Cậu bạn to khỏe yêu ca hát",
                "search_query": "Takeshi Gian Goda Doraemon anime",
                "keywords": ["gian", "chaien", "takeshi"]
            }
        ]
    },

    "attack_on_titan": {
        "id": "attack_on_titan",
        "title": "Attack on Titan (Đại Chiến Titan)",
        "original_title": "進撃の巨人",
        "creator": "Hajime Isayama (Nhật Bản)",
        "category": "Anime Sử Thi & Giả Tưởng Đen Tối (Dark Fantasy)",
        "icon": "⚔️",
        "badge": "Kiệt Tác Anime Hiện Đại",
        "color_theme": "emerald",
        "description": "Kiệt tác đen tối và bi tráng của Hajime Isayama kể về cuộc chiến sinh tồn khắc nghiệt của nhân loại sau những bức tường trước loài khổng lồ Titan ăn thịt người.",
        "search_enrichment": "Attack on Titan Shingeki no Kyojin anime Hajime Isayama Survey Corps wallpaper",
        "keywords": [
            "attack on titan", "aot", "shingeki no kyojin", "dai chien titan",
            "eren", "eren yeager", "mikasa", "mikasa ackerman", "levi", "levi ackerman",
            "armin", "armin arlert", "trinh sat doan", "titan dai hinh", "titan thiet giap"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Attack on Titan",
                "vietnamese_name": "Quân Trinh Sát & Thế Giới Titan",
                "icon": "⚔️",
                "badge": "Toàn Bộ Universe",
                "search_query": "Attack on Titan Survey Corps anime wallpaper",
                "keywords": ["all", "tat ca", "attack on titan"]
            },
            {
                "id": "eren",
                "name": "Eren Yeager (Titan Tiến Công & Thủy Tổ)",
                "vietnamese_name": "Eren Yeager",
                "icon": "🕊️",
                "badge": "Titan Thủy Tổ · Tự do tối thượng",
                "search_query": "Eren Yeager Attack on Titan Founding Titan anime",
                "keywords": ["eren", "eren yeager", "titan tien cong"]
            },
            {
                "id": "levi",
                "name": "Levi Ackerman (Chiến Binh Mạnh Nhất)",
                "vietnamese_name": "Đại Đội Trưởng Levi Ackerman",
                "icon": "⚔️",
                "badge": "Chiến binh mạnh nhất nhân loại",
                "search_query": "Levi Ackerman Attack on Titan ODM gear anime wallpaper",
                "keywords": ["levi", "levi ackerman"]
            },
            {
                "id": "mikasa",
                "name": "Mikasa Ackerman",
                "vietnamese_name": "Mikasa Ackerman (Chiếc Khăn Đỏ)",
                "icon": "🧣",
                "badge": "Nữ chiến binh tinh nhuệ dòng máu Ackerman",
                "search_query": "Mikasa Ackerman Attack on Titan red scarf anime",
                "keywords": ["mikasa", "mikasa ackerman"]
            }
        ]
    },

    "demon_slayer": {
        "id": "demon_slayer",
        "title": "Kimetsu no Yaiba (Thanh Gươm Diệt Quỷ)",
        "original_title": "鬼滅の刃",
        "creator": "Koyoharu Gotouge (Nhật Bản)",
        "category": "Anime Kiếm Hiệp Đồ Họa Đỉnh Cao · Ufotable",
        "icon": "🗡️",
        "badge": "Hiện Tượng Anime Toàn Cầu",
        "color_theme": "purple",
        "description": "Hành trình diệt quỷ đầy xúc động của Tanjiro Kamado nhằm tìm phương thuốc chữa cho em gái Nezuko bị biến thành quỷ, với đồ họa đỉnh cao từ xưởng Ufotable.",
        "search_enrichment": "Kimetsu no Yaiba Demon Slayer anime Ufotable Tanjirou Nezuko Hashira wallpaper",
        "keywords": [
            "kimetsu no yaiba", "demon slayer", "thanh guom diet quy", "diet quy",
            "tanjiro", "tanjiro kamado", "nezuko", "nezuko kamado", "zenitsu",
            "inosuke", "rengoku", "kyojuro rengoku", "giyu", "giyu tomioka",
            "shinobu", "tru cot", "hashira", "muzan"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Demon Slayer",
                "vietnamese_name": "Sát Quỷ Đội & Các Trụ Cột",
                "icon": "🗡️",
                "badge": "Toàn Bộ Universe",
                "search_query": "Kimetsu no Yaiba Demon Slayer Hashira anime wallpaper",
                "keywords": ["all", "tat ca", "demon slayer"]
            },
            {
                "id": "tanjiro",
                "name": "Kamado Tanjiro (Hơi Thở Mặt Trời)",
                "vietnamese_name": "Kamado Tanjiro",
                "icon": "🌊",
                "badge": "Kiếm sĩ Hơi Thở Mặt Trời (Hinokami Kagura)",
                "search_query": "Kamado Tanjiro Hinokami Kagura Demon Slayer anime wallpaper",
                "keywords": ["tanjiro", "kamado tanjiro", "hinokami kagura"]
            },
            {
                "id": "nezuko",
                "name": "Kamado Nezuko (Quỷ Hoa)",
                "vietnamese_name": "Em Gái Quỷ Nezuko",
                "icon": "🌸",
                "badge": "Cô bé quỷ bảo vệ con người",
                "search_query": "Kamado Nezuko Demon Slayer cute anime wallpaper",
                "keywords": ["nezuko", "kamado nezuko"]
            },
            {
                "id": "rengoku",
                "name": "Kyojuro Rengoku (Viêm Trụ)",
                "vietnamese_name": "Viêm Trụ Rengoku Kyojuro",
                "icon": "🔥",
                "badge": "Ngọn lửa kiêu hùng · Chuyến tàu vô tận",
                "search_query": "Kyojuro Rengoku Flame Hashira Demon Slayer anime wallpaper",
                "keywords": ["rengoku", "viem tru", "kyojuro rengoku"]
            },
            {
                "id": "giyu",
                "name": "Giyu Tomioka (Thủy Trụ)",
                "vietnamese_name": "Thủy Trụ Tomioka Giyu",
                "icon": "💧",
                "badge": "Thủy Trụ điềm tĩnh · Hơi Thở Của Nước",
                "search_query": "Giyu Tomioka Water Hashira Demon Slayer anime wallpaper",
                "keywords": ["giyu", "thuy tru", "tomioka giyu"]
            }
        ]
    },

    "pokemon": {
        "id": "pokemon",
        "title": "Pokémon (Bảo Bối Thần Kỳ)",
        "original_title": "ポケットモンスター",
        "creator": "Satoshi Tajiri & Nintendo",
        "category": "Thương Hiệu Giải Trí Đắt Giá Nhất Hành Tinh",
        "icon": "⚡",
        "badge": "Huyền Thoại Toàn Cầu",
        "color_theme": "yellow",
        "description": "Vũ trụ Pokémon đầy kỳ thú nơi các nhà huấn luyện cùng những sinh vật huyền diệu phiêu lưu, kết bạn và thi đấu để trở thành Nhà Vô Địch Thế Giới.",
        "search_enrichment": "Pokemon Pikachu Nintendo anime official art characters wallpaper",
        "keywords": [
            "pokemon", "pokémon", "bao boi than ky", "pikachu", "charizard",
            "mewtwo", "satoshi", "ash ketchum", "lucario", "eevee", "gengar", "rayquaza"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Pokémon",
                "vietnamese_name": "Thế Giới Pokémon Đa Sắc",
                "icon": "⚡",
                "badge": "Toàn Bộ Universe",
                "search_query": "Pokemon anime Pikachu Charizard wallpaper",
                "keywords": ["all", "tat ca", "pokemon"]
            },
            {
                "id": "pikachu",
                "name": "Pikachu (Hệ Điện)",
                "vietnamese_name": "Pikachu Đáng Yêu",
                "icon": "⚡",
                "badge": "Biểu tượng Pokémon toàn cầu · Điện 100,000 Volt",
                "search_query": "Pikachu cute electric Pokemon anime wallpaper",
                "keywords": ["pikachu", "chuot dien"]
            },
            {
                "id": "charizard",
                "name": "Charizard (Rồng Lửa)",
                "vietnamese_name": "Rồng Lửa Charizard",
                "icon": "🔥",
                "badge": "Rồng lửa hùng dũng hệ Lửa/Bay",
                "search_query": "Charizard fire Pokemon anime official art wallpaper",
                "keywords": ["charizard", "rong lua charizard"]
            },
            {
                "id": "mewtwo",
                "name": "Mewtwo (Siêu Pokémon Nhân Tạo)",
                "vietnamese_name": "Mewtwo Huyền Thoại",
                "icon": "🔮",
                "badge": "Pokémon huyền thoại mang năng lực tâm linh tối thượng",
                "search_query": "Mewtwo legendary psychic Pokemon anime wallpaper",
                "keywords": ["mewtwo", "mew two"]
            }
        ]
    },

    "marvel": {
        "id": "marvel",
        "title": "Marvel Avengers & Vũ Trụ Siêu Anh Hùng",
        "original_title": "Marvel Comics / MCU",
        "creator": "Stan Lee, Jack Kirby & Marvel",
        "category": "Vũ Trụ Siêu Anh Hùng Điện Ảnh & Truyện Tranh",
        "icon": "🛡️",
        "badge": "Vũ Trụ Điện Ảnh Marvel",
        "color_theme": "red",
        "description": "Vũ trụ siêu anh hùng vĩ đại nhất của hãng Marvel với biệt đội Avengers, Spider-Man, Iron Man bảo vệ Trái Đất và đa vũ trụ.",
        "search_enrichment": "Marvel Avengers MCU Stan Lee superhero official poster wallpaper",
        "keywords": [
            "marvel", "avengers", "mcu", "sieu anh hung marvel", "iron man", "tony stark",
            "spider man", "spiderman", "nguoi nhen", "captain america", "thor", "than sam",
            "hulk", "doctor strange", "thanos", "loki", "wolverine"
        ],
        "characters": [
            {
                "id": "all",
                "name": "Tất Cả Marvel Avengers",
                "vietnamese_name": "Biệt Đội Siêu Anh Hùng Avengers",
                "icon": "🛡️",
                "badge": "Toàn Bộ Universe",
                "search_query": "Marvel Avengers superhero movie poster wallpaper",
                "keywords": ["all", "tat ca", "marvel"]
            },
            {
                "id": "iron_man",
                "name": "Iron Man (Tony Stark)",
                "vietnamese_name": "Người Sắt Iron Man",
                "icon": "🦾",
                "badge": "Tỷ phú công nghệ · Bộ giáp Nano",
                "search_query": "Iron Man Tony Stark Marvel MCU wallpaper",
                "keywords": ["iron man", "tony stark", "nguoi sat"]
            },
            {
                "id": "spider_man",
                "name": "Spider-Man (Người Nhện)",
                "vietnamese_name": "Người Nhện Thân Thiện (Peter Parker)",
                "icon": "🕷️",
                "badge": "Người Nhện thân thiện của khu phố",
                "search_query": "Spider-Man Peter Parker Marvel comics wallpaper",
                "keywords": ["spider man", "spiderman", "nguoi nhen", "peter parker"]
            },
            {
                "id": "thor",
                "name": "Thor (Thần Sấm)",
                "vietnamese_name": "Thần Sấm Thor (Asgard)",
                "icon": "⚡",
                "badge": "Hoàng tử Asgard · Búa Mjolnir & Stormbreaker",
                "search_query": "Thor God of Thunder Marvel lightning wallpaper",
                "keywords": ["thor", "than sam", "mjolnir"]
            },
            {
                "id": "captain_america",
                "name": "Captain America (Steve Rogers)",
                "vietnamese_name": "Đội Trưởng Mỹ",
                "icon": "⭐",
                "badge": "Siêu chiến binh với chiếc khiên Vibranium",
                "search_query": "Captain America Steve Rogers shield Marvel wallpaper",
                "keywords": ["captain america", "steve rogers", "doi truong my"]
            }
        ]
    }
}


def detect_franchise_entity(query: str) -> Optional[Dict[str, Any]]:
    """
    Analyzes user search query to detect if it matches any pop-culture franchise,
    anime universe, or specific character. Returns franchise info with highlighted
    character chip if matched.
    """
    if not query:
        return None

    q_norm = normalize_vietnamese_text(query)
    q_lower = query.lower().strip()

    best_match = None
    best_char_id = "all"
    highest_score = 0

    for f_id, f_data in FRANCHISE_KNOWLEDGE_BASE.items():
        # Check franchise keywords
        for kw in f_data["keywords"]:
            kw_norm = normalize_vietnamese_text(kw)
            if kw_norm == q_norm:
                score = 100
            elif kw_norm in q_norm or q_norm in kw_norm:
                score = 80
            else:
                score = 0

            if score > highest_score:
                highest_score = score
                best_match = f_data

        # Check character keywords within this franchise
        for ch in f_data["characters"]:
            for ch_kw in ch["keywords"]:
                ch_norm = normalize_vietnamese_text(ch_kw)
                if ch_norm == q_norm:
                    score = 95
                    best_char_id = ch["id"]
                elif ch_norm in q_norm:
                    score = 85
                    best_char_id = ch["id"]
                else:
                    score = 0

                if score > highest_score:
                    highest_score = score
                    best_match = f_data

    if not best_match or highest_score < 60:
        return None

    # Clone structure and mark active character
    entity_result = {
        "id": best_match["id"],
        "title": best_match["title"],
        "original_title": best_match["original_title"],
        "creator": best_match["creator"],
        "category": best_match["category"],
        "icon": best_match["icon"],
        "badge": best_match["badge"],
        "color_theme": best_match.get("color_theme", "cyan"),
        "description": best_match["description"],
        "search_enrichment": best_match["search_enrichment"],
        "active_character_id": best_char_id,
        "characters": [
            {
                "id": ch["id"],
                "name": ch["name"],
                "vietnamese_name": ch["vietnamese_name"],
                "icon": ch["icon"],
                "badge": ch["badge"],
                "search_query": ch["search_query"],
                "is_active": (ch["id"] == best_char_id)
            }
            for ch in best_match["characters"]
        ]
    }

    return entity_result
