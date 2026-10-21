# Registration day + CCTV test - checklist

30+ students register karne aur real CCTV / recorded video par test karne ke liye.

## 0. Shuru karne se pehle (5 min)

```bash
.venv/Scripts/python.exe -m backend.main --config configs/config.yaml --no-window --dashboard
```

```bash
cd frontend/web && npm run dev
```

- Registration **usi laptop par** karo jisme webcam hai, aur browser mein
  `http://localhost:5173/portal/faces` kholo. Camera sirf `localhost` ya
  `https://` par chalta hai - LAN IP (`http://192.168...`) par browser camera
  block karta hai (tab "Upload photos" use karo, ya https tunnel).
- Face Registry page par "Models: ready" dikhna chahiye.
- **Backup:** pehle `data/faces/faces.db` ki copy rakh lo, aur registration ke
  baad dobara copy karo. Saare students isi ek file mein hain.

## 1. Har student (~1 minute)

1. **Add user** - Full name, ID / Roll no., Role (Student), Class.
   - Naam unique hona chahiye. Do students ka same naam ho to
     `Aarav Sharma (R-101)` likho.
   - Naam English letters mein (video par Hindi akshar `?` dikhenge).
2. Save karte hi **Face enrolment** khulta hai -> **Start camera**.
3. Student ko ek-ek karke bolo, har pose par 1-2 second **ruke**:
   - seedha camera dekho (Front)
   - sir thoda left ghumao (~30-45 degree, dono aankhein dikhni chahiye)
   - sir thoda right ghumao
   - (optional) chin upar, chin neeche
   Box **hara** hote hi photo apne aap le li jaati hai. Jo bhi zaroori pose
   dikhe woh le liya jaata hai - order zaroori nahi.
4. "Enrolled - front, left and right are captured" dikhe to **Done**.

Achhi capture ke liye: chehre par saamne se roshni (peeche window nahi), frame
mein sirf ek insaan, cap/mask hata do, chashma chalega.

Agar mana kare to message padho - woh batata hai kya theek karna hai
("Move closer", "blurry - hold still", "Too dark", "turned too far").
"This face is already registered as X" = yeh student pehle se registered hai.

## 2. Registration ke baad check (2 min)

Face Registry page ke neeche **Check recognition**:

- **Estimate accuracy** dabao. "False accept rate" 0% hona chahiye. Agar
  "suggested threshold" bahut alag aaye to bata dena.
- **Test a photo**: kisi student ki nayi photo do - sahi naam aana chahiye.

## 3. CCTV / recorded video par test

1. **Cameras** page -> camera add karo:
   - CCTV: source `RTSP`, address `rtsp://user:pass@IP:554/...`
   - Recorded video: source `Video file`, upload karo, "Play again" on rakho.
2. Us camera ke AI modules mein yeh on karo: Person detection + tracking, Face
   recognition, Unknown-person alerts, Entry / exit gate counter
   (Face liveness optional).
3. **Live Monitoring**: pehchane gaye student ka naam hare box par dikhta hai
   (chehre par aur uske body box par `#3 Aarav Sharma`). Anjaan = narangi
   "Unknown". Chhota/dhundla chehra = grey "Face" (check hi nahi kiya).
4. **Reports**: entry/exit log (naam ke saath), unknown logon ki snapshot,
   **Export CSV** aur **Attendance CSV**.

### CCTV par sabse zaroori baat: chehre ka size

Recognition tabhi hota hai jab video mein chehra **kam se kam 60 pixel** ka ho.
1080p camera par iska matlab hai insaan camera se lagbhag 4-5 meter ke andar.
Agar zyadatar chehre grey "Face" dikh rahe hain:

- Camera ko darwaze/gate ke paas aur aankh ki oonchai ke kareeb lagao (bahut
  upar se sirf sar ki chhat dikhti hai), ya
- Settings page par `face_recognition -> min_face_size` 60 se 40 karo (zyada
  log check honge, lekin galti thodi badhegi).

### Accuracy aur badhani ho

Webcam se register kiya chehra CCTV par thoda alag dikhta hai (angle, roshni,
quality). Har student ki CCTV se 2-3 saaf snapshot/photo bhi **Upload photos**
se add kar do - CCTV par pehchaan kaafi behtar hogi.

### Gate line

Entry/exit ke liye `configs/config.yaml` mein `gate_counter.line` ko apne
camera ke darwaze par set karo (0-1 fractions, e.g. `[[0.1, 0.7], [0.9, 0.7]]`)
aur `inside: left/right`. Entry/exit ulta aaye to `inside` badal do. Line sab
cameras ke liye ek hi hai.

## 4. Jaanne layak limits

- Liveness abhi sirf **flag** karta hai (photo dikhane par alert), block nahi.
- YouTube camera ko cookies chahiye (YouTube bot-check), warna "failed".
- Har camera alag process mein ~1-1.5 GB RAM leta hai. PC slow ho to cameras
  kam karo ya `process_per_camera: false`.

## 5. Kuch gadbad ho to

| Dikkat | Kya karo |
|---|---|
| Camera start nahi hota | `localhost` par kholo, browser ko camera permission do, ya Upload photos |
| Box hara nahi hota | Message padho: paas aao / roshni / ek hi insaan / sir kam ghumao |
| "already registered as X" | Student pehle se hai - list mein search karo |
| CCTV par sab "Face" (grey) | Chehre bahut chhote - section 3 dekho |
| Sab cameras "STALLED" | Backend terminal dekho; backend restart karo |
| Galat naam aa raha hai | Reports mein snapshot dekho; us student ki 2-3 aur captures lo |
