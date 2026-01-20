# DIY Webcam Card Scanner Setup

A simple, effective card scanning station using everyday household objects.

## What You'll Need

### Required:
- **Webcam** (built-in laptop webcam or USB webcam)
- **Books or boxes** (for elevation)
- **White paper or poster board** (for background)
- **Desk lamp** or good room lighting

### Optional (Recommended):
- **Phone stand/tripod** (can improvise with books)
- **Cardboard box** (to create a light box)
- **White tissue paper** (for diffusing harsh light)

## Setup Instructions

### Option 1: Simple Desktop Setup (5 minutes)

```
┌─────────────────────────────────────┐
│          WEBCAM                      │
│            ↓                         │
│    [Books/Box Stack]                 │
│                                      │
│    ┌──────────────┐                 │
│    │  White Paper │  ← Desk Surface │
│    │   [CARD]     │                 │
│    └──────────────┘                 │
│                                      │
│  💡 Lamp ←                          │
└─────────────────────────────────────┘
```

**Steps:**
1. Clear a space on your desk
2. Place white paper/poster board on desk surface
3. Stack books or boxes to elevate webcam 12-18 inches above desk
4. Position webcam pointing straight down (90° angle)
5. Ensure even lighting from desk lamp(s) on both sides
6. Avoid shadows and glare

**Key Points:**
- Camera should be directly overhead, not at an angle
- Background should be plain white or light colored
- Avoid harsh shadows by using multiple light sources
- Remove card from sleeve if possible (reduces glare)

### Option 2: DIY Light Box (15 minutes)

For professional-quality results, create a simple light box:

**Materials:**
- Large cardboard box (shoebox size or bigger)
- White paper/cloth for background
- Desk lamps (2 if possible)
- Scissors/box cutter
- Tape

**Instructions:**
1. Cut a hole in the top of box for webcam to point through
2. Cut windows in the sides for lighting
3. Line inside with white paper/cloth
4. Place card on white background at bottom
5. Shine lamps through side windows
6. Position webcam in top hole pointing straight down

**Benefits:**
- Eliminates shadows
- Consistent lighting
- Diffused light for better OCR
- Reduces glare from card sleeves

## Webcam Positioning

### Height
- **Optimal:** 12-18 inches above card
- **Minimum:** 8 inches
- **Maximum:** 24 inches (may lose detail)

### Angle
- **CRITICAL:** 90° (perpendicular to card surface)
- NOT at an angle - must be directly overhead
- Use a ruler or level to check if needed

### Focus
- Test capture a few images first
- Adjust webcam focus if needed (some webcams have manual focus)
- Card text should be sharp and readable

## Lighting Tips

### Best Practices:
- ✓ Two light sources (one on each side) - eliminates shadows
- ✓ Diffused light (lamp with shade, or tissue paper over light)
- ✓ Position lights at 45° angles to card
- ✓ Daylight/natural light is excellent (near window)

### Avoid:
- ✗ Single light source directly overhead (creates harsh shadows)
- ✗ Flash photography
- ✗ Backlit setup (light behind card)
- ✗ Scanning cards in sleeves when possible (causes glare)

## Testing Your Setup

Run the webcam capture script:
```bash
python webcam_capture.py
```

Look for:
- Card fills most of frame
- Text is sharp and readable
- No harsh shadows
- Even lighting across entire card
- No glare or reflections

## Troubleshooting

### Card Not Detected
- Ensure white/light background
- Check that card is flat and fully visible
- Adjust lighting to increase contrast

### Blurry Images
- Reduce camera height
- Check webcam focus
- Ensure card is completely flat
- Hold still during capture

### OCR Not Reading Text
- Increase lighting
- Remove card from sleeve
- Clean card surface
- Ensure camera is directly overhead (not angled)

### Shadows
- Add second light source on opposite side
- Move light sources further from card
- Use diffused lighting (lamp shade, tissue paper)

## Quick Tips

1. **Consistency is Key:** Once you find a good setup, mark the positions with tape
2. **Batch Processing:** Scan multiple cards in one session
3. **Card Sleeves:** Remove if possible, or use matte sleeves (not glossy)
4. **Background:** White printer paper works perfectly
5. **Stability:** Make sure webcam mount is stable (not wobbling)

## Advanced Setup Ideas

### Motorized Turntable
- Place lazy susan or turntable under card
- Rotate to capture multiple angles
- Good for foil/holographic cards

### Multi-Camera Setup
- Use multiple webcams for different angles
- Capture front and back simultaneously
- Use phone camera as second camera

### Foot Pedal Trigger
- Use USB foot pedal as keyboard input
- Hands-free capture while handling cards
- Speeds up batch scanning

---

**Next Step:** Run `python webcam_capture.py` to start capturing cards!
