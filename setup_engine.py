"""Build empty FL projects using the installed FL Empty template and FL presets.

Unknown template events and opaque plugin state are retained without deserialization.
No third-party packages or FL installation modifications are required.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import struct
import tempfile

APP_VERSION = '1.3'
MAX_CHANNELS = 125


def discover_installations():
    candidates = []
    for drive in ('C:', 'D:', 'E:', 'F:'):
        for parent in ('Program Files', 'Program Files (x86)'):
            root = Path(drive + '/') / parent / 'Image-Line'
            if root.is_dir():
                candidates.extend(p for p in root.glob('FL Studio*') if (p / 'FL64.exe').is_file())
    return [str(p) for p in sorted(set(candidates), key=lambda p: p.name, reverse=True)]


def user_data_folder():
    # FL can redirect Documents (including OneDrive) or its own user-data folder.
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Image-Line\FL Studio') as key:
            for field in ('User data folder', 'UserDataPath'):
                try:
                    value, _ = winreg.QueryValueEx(key, field)
                    p = Path(os.path.expandvars(value))
                    if p.is_dir():
                        return p
                except OSError:
                    pass
    except (ImportError, OSError):
        pass
    documents = Path.home() / 'Documents'
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as key:
            documents = Path(os.path.expandvars(winreg.QueryValueEx(key, 'Personal')[0]))
    except (ImportError, OSError):
        pass
    return documents / 'Image-Line' / 'FL Studio'


def scan_generators(fl_dir):
    roots = [user_data_folder() / 'Presets' / 'Plugin database' / 'Installed' / 'Generators',
             Path(fl_dir) / 'Data' / 'Patches' / 'Plugin database' / 'Installed' / 'Generators']
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob('*.fst')):
            if path.stem in {'Fruity Wrapper', 'FL Studio VSTi', 'FL Studio VSTi (Multi)'}:
                continue
            kind = path.parent.name
            name = f'{path.stem} ({kind})'
            found.setdefault(name.casefold(), {'name': name, 'path': str(path), 'kind': kind})
    return sorted(found.values(), key=lambda item: item['name'].casefold())


def read_fl(path):
    data = Path(path).read_bytes()
    if len(data) < 22 or data[:4] != b'FLhd':
        raise ValueError(f'{Path(path).name} is not an FL Studio project or preset.')
    n = struct.unpack_from('<I', data, 4)[0]
    if n != 6:
        raise ValueError('This FL file uses an unsupported header.')
    header = data[8:14]
    if data[14:18] != b'FLdt':
        raise ValueError('This FL file has no supported event stream.')
    end = 22 + struct.unpack_from('<I', data, 18)[0]
    if end != len(data):
        raise ValueError('This FL file is incomplete or uses an unsupported container.')
    pos = 22
    events = []
    while pos < end:
        event_id = data[pos]
        pos += 1
        payload_start = pos
        extended = event_id == 172
        if extended:
            if pos + 4 > end:
                raise ValueError('Truncated extended FL event.')
            descriptor = struct.unpack_from('<I', data, pos)[0]
            event_type = descriptor >> 24
            pos += 4
            if event_type not in (0x00, 0x40, 0x80, 0xC0):
                raise ValueError('Unsupported extended FL event type.')
        else:
            event_type = event_id
        if event_type < 192:
            length = (1, 2, 4)[event_type // 64]
        else:
            length = shift = 0
            while True:
                if pos >= end or shift > 28:
                    raise ValueError('Invalid FL event length.')
                byte = data[pos]
                pos += 1
                length |= (byte & 127) << shift
                shift += 7
                if not byte & 128:
                    break
        if pos + length > end:
            raise ValueError('Truncated FL event.')
        events.append((event_id, data[payload_start if extended else pos:pos + length]))
        pos += length
    return header, events


def encode_fl(header, events):
    stream = bytearray()
    for event_id, payload in events:
        stream.append(event_id)
        if event_id == 172:
            # Extended events include their descriptor and any length prefix.
            if len(payload) < 5:
                raise ValueError('Invalid extended FL event.')
        elif event_id < 192:
            if len(payload) != (1, 2, 4)[event_id // 64]:
                raise ValueError(f'Invalid fixed FL event {event_id}.')
        else:
            n = len(payload)
            while n >= 128:
                stream.append((n & 127) | 128)
                n >>= 7
            stream.append(n)
        stream.extend(payload)
    return b'FLhd' + struct.pack('<I', len(header)) + header + b'FLdt' + struct.pack('<I', len(stream)) + stream


def _text(value):
    return (value + '\0').encode('utf-16-le')


def _replace(events, event_id, payload, *, required=False):
    indices = [i for i, (key, _) in enumerate(events) if key == event_id]
    if not indices:
        if required:
            raise ValueError(f'Template is missing required event {event_id}.')
        events.append((event_id, payload))
    else:
        for i in indices:
            events[i] = (event_id, payload)


def _number(value, label, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f'{label} must be a number.')
    try:
        n = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{label} must be a number.') from None
    if not math.isfinite(n) or not minimum <= n <= maximum:
        raise ValueError(f'{label} must be between {minimum} and {maximum}.')
    return n


def _clean_name(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError(f'{label} needs a name of 1 to 100 characters.')
    if any(ord(c) < 32 for c in value):
        raise ValueError(f'{label} cannot contain control characters.')
    return value.strip()


def _color(value):
    if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
        raise ValueError('Channel color must be a six-digit hex color.')
    return bytes.fromhex(value[1:]) + b'\0'


def _template(fl_dir):
    path = Path(fl_dir) / 'Data' / 'Templates' / 'Empty' / 'Empty.flp'
    if not path.is_file():
        raise ValueError('Choose the FL Studio installation folder containing FL64.exe and Data/Templates/Empty/Empty.flp.')
    header, events = read_fl(path)
    channel_starts = [i for i, (key, _) in enumerate(events) if key == 64]
    if struct.unpack_from('<H', header, 0)[0] != 0 or struct.unpack_from('<H', header, 2)[0] != 1 or len(channel_starts) != 1:
        raise ValueError('The installed Empty template has an unsupported channel layout. It was left unchanged.')
    start = channel_starts[0]
    finish = next((i for i in range(start + 1, len(events)) if events[i][0] == 99), None)
    if finish is None:
        raise ValueError('The Empty template has an unsupported arrangement layout.')
    channel = events[start:finish]
    if dict(channel).get(21) != b'\0' or len([1 for k, _ in channel if k == 218]) != 5:
        raise ValueError('The Empty template does not contain the expected Sampler.')
    if any(k in {223, 224} and v for k, v in events):
        raise ValueError('The Empty template unexpectedly contains musical data.')
    return header, events[:start], channel, events[finish:]


def _preset_channel(base, path):
    if not path or not Path(path).is_file() or Path(path).suffix.lower() != '.fst':
        raise ValueError('Choose an installed synth or a saved FL Studio .fst preset.')
    header, preset = read_fl(path)
    fmt = struct.unpack_from('<H', header)[0]
    if fmt not in (0x20, 0x30):
        raise ValueError(f'{Path(path).name} is not a supported instrument/channel preset.')
    for key, value in preset:
        if key == 172 and struct.unpack_from('<I', value)[0] not in {0xC0000101, 0x00000100}:
            raise ValueError('This preset has unsupported extended settings. Save a simple generator .fst preset.')
    for event_id in (201, 212, 213):
        if sum(k == event_id for k, _ in preset) > 1:
            raise ValueError('This preset contains multiple plugin definitions. Save one synth as a fresh .fst preset.')
    # Older FSTs use ANSI strings; avoid silently importing them as Unicode.
    internal_name = next((v for k, v in preset if k == 201), b'')
    if internal_name and (len(internal_name) % 2 or not internal_name.endswith(b'\0\0')):
        raise ValueError('This is a legacy FL preset. Open it in your current FL Studio and save a new .fst preset first.')
    if sum(k == 64 for k, _ in preset) > 1:
        raise ValueError('Multi-channel presets are not supported. Save one instrument as an .fst preset.')
    if not any(k == 201 and v not in (b'', b'\0\0') for k, v in preset):
        raise ValueError('This preset does not identify a synth. Use a generator .fst preset.')
    if any(k == 21 and v != b'\x02' for k, v in preset):
        raise ValueError('This is not a synth channel preset.')
    # FST event payloads, including opaque VST/native state, are copied intact.
    plugin_ids = {201, 212, 213, 203, 155, 128, 41}
    preset_metadata = {199, 159, 169, 28, 37, 172}
    if fmt == 0x30:
        if any(k not in plugin_ids | preset_metadata for k, _ in preset):
            raise ValueError('This plugin preset has unsupported extra settings. Save it as a fresh generator .fst preset in FL Studio.')
        allowed = [(k, v) for k, v in preset if k in plugin_ids]
    else:
        # Only single-channel records are accepted; never import controller,
        # mixer, pattern or playlist links belonging to another project.
        valid = {0, 2, 3, 15, 20, 21, 22, 32, 41, 48, 50, 69, 70, 71,
                 72, 73, 74, 75, 76, 83, 85, 86, 89, 97, 104, 131, 132,
                 135, 138, 139, 140, 142, 143, 144, 145, 153, 192, 196,
                 209, 215, 218, 219, 221, 228, 229} | plugin_ids
        unknown = {k for k, _ in preset} - valid - {64} - preset_metadata
        if unknown:
            raise ValueError(f'This channel preset includes unsupported links/settings ({sorted(unknown)}). Save the synth as a plugin preset instead.')
        allowed = [(k, v) for k, v in preset if k in valid]
    override_ids = {k for k, _ in allowed}
    remaining = [(k, v) for k, v in base if k not in override_ids and k not in {196, 192}]
    # Place plugin records immediately after ChannelID.Type, as FL saves them.
    plugin = [(k, v) for k, v in allowed if k in plugin_ids]
    settings = [(k, v) for k, v in allowed if k not in plugin_ids and k != 21]
    idx = next(i for i, (k, _) in enumerate(remaining) if k == 64) + 1
    remaining = [(k, v) for k, v in remaining if k != 21]
    remaining[idx:idx] = [(21, b'\x02')] + plugin
    remaining.extend(settings)
    _replace(remaining, 21, b'\x02')
    return remaining


def _make_channel(base, row, index):
    if not isinstance(row, dict):
        raise ValueError('Each channel must be a settings object.')
    kind = row.get('kind')
    if kind not in ('sampler', 'synth'):
        raise ValueError('Choose Sampler or Synth for each channel.')
    name = _clean_name(row.get('name', f'Channel {index + 1}'), 'Channel')
    channel = list(base)
    if kind == 'synth':
        channel = _preset_channel(channel, row.get('preset_path', ''))
    _replace(channel, 64, struct.pack('<H', index), required=True)
    _replace(channel, 203, _text(name))
    _replace(channel, 128, _color(row.get('color', '#719FB5')))
    _replace(channel, 41, b'\x01')  # keep the chosen color instead of FL's auto color
    # One unsorted channel group; never keep old preset group/cut links.
    _replace(channel, 145, struct.pack('<i', 0), required=True)
    _replace(channel, 132, b'\0' * 4)
    route = _number(row.get('mixer', index + 1), f'{name}: mixer track', 0, 125)
    if route != int(route):
        raise ValueError('Mixer track must be a whole number.')
    _replace(channel, 104, struct.pack('<H', int(route)))
    channel = [(k, v) for k, v in channel if k not in {22, 2, 3, 72, 73}]
    volume = _number(row.get('volume', 78.125), f'{name}: volume', 0, 100)
    pan = _number(row.get('pan', 0), f'{name}: pan', -100, 100)
    levels = bytearray(next(v for k, v in channel if k == 219))
    struct.pack_into('<iI', levels, 0, round((pan + 100) * 64), round(volume * 128))
    _replace(channel, 219, bytes(levels), required=True)
    if kind == 'sampler':
        sample = row.get('sample_path', '')
        if sample:
            p = Path(sample)
            if not p.is_file():
                raise ValueError(f'{name}: sample file is missing: {sample}')
            if p.suffix.lower() not in {'.wav', '.aif', '.aiff', '.mp3', '.ogg', '.flac'}:
                raise ValueError(f'{name}: choose a WAV, AIFF, MP3, OGG or FLAC sample.')
            _replace(channel, 196, _text(str(p.resolve())))
        env = row.get('envelope', {})
        if not isinstance(env, dict):
            raise ValueError('Sampler envelope settings must be an object.')
        enabled = env.get('enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError('Envelope enabled must be true or false.')
        env_positions = [i for i, (k, _) in enumerate(channel) if k == 218]
        if len(env_positions) != 5:
            raise ValueError('The sampler template has an unsupported envelope layout.')
        pos = env_positions[1]
        payload = bytearray(channel[pos][1])
        struct.pack_into('<i', payload, 4, int(enabled))
        struct.pack_into('<i', payload, 8, 100)  # no envelope pre-delay
        for field, offset, default in [('attack', 12, 0), ('hold', 16, 0), ('decay', 20, 25), ('release', 28, 5)]:
            n = _number(env.get(field, default), f'{name}: {field}', 0, 100)
            struct.pack_into('<i', payload, offset, round(100 + n * (65536 - 100) / 100))
        sustain = _number(env.get('sustain', 100), f'{name}: sustain', 0, 100)
        struct.pack_into('<i', payload, 24, round(sustain * 128 / 100))
        channel[pos] = (218, bytes(payload))
    return channel


def _configure_mixer(suffix, rows):
    """Extend the stock empty mixer and label the requested destinations."""
    count_pos = next((i for i, (k, _) in enumerate(suffix) if k == 103), None)
    params_pos = next((i for i, (k, _) in enumerate(suffix) if k == 225), None)
    if count_pos is None or params_pos is None:
        raise ValueError('The Empty template has no supported mixer layout.')
    block = []
    blocks = []
    for event in suffix[count_pos + 1:params_pos]:
        block.append(event)
        if event[0] == 147:
            blocks.append(block)
            block = []
    stored_count = int.from_bytes(suffix[count_pos][1], 'little')
    if block or len(blocks) != stored_count or len(blocks) < 3:
        raise ValueError('The Empty template has an unsupported mixer layout.')
    highest = max(int(row.get('mixer', i + 1)) for i, row in enumerate(rows))
    existing = len(blocks) - 2  # master + normal inserts + current insert
    if highest > existing:
        normal = list(blocks[1])
        blocks[-1:-1] = [list(normal) for _ in range(highest - existing)]
    by_route = {}
    for i, row in enumerate(rows):
        route = int(row.get('mixer', i + 1))
        if route:
            by_route.setdefault(route, []).append(row)
    for route, group in by_route.items():
        block = blocks[route]
        pos = next(i for i, (k, _) in enumerate(block) if k == 236)
        name = ' + '.join(row.get('name', 'Channel') for row in group)[:100]
        block[pos:pos] = [(149, _color(group[0].get('color', '#719FB5'))), (204, _text(name))]
    return (suffix[:count_pos] + [(103, struct.pack('<H', len(blocks)))]
            + [event for block in blocks for event in block] + suffix[params_pos:])


def build_project(recipe, output_path, fl_dir):
    if not isinstance(recipe, dict):
        raise ValueError('Setup must be a settings object.')
    title = _clean_name(recipe.get('title', 'My FL Setup'), 'Project')
    bpm = _number(recipe.get('bpm', 140), 'BPM', 10, 522)
    rows = recipe.get('channels')
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CHANNELS:
        raise ValueError(f'Add between 1 and {MAX_CHANNELS} channels.')
    dest = Path(output_path)
    if dest.suffix.lower() != '.flp':
        raise ValueError('Save the project with an .flp extension.')
    if not dest.parent.is_dir():
        raise ValueError('The output folder does not exist.')
    # Never overwrite a preset/template source, even if a caller bypasses the UI.
    protected = [Path(fl_dir) / 'Data' / 'Templates' / 'Empty' / 'Empty.flp']
    protected += [Path(r[k]) for r in rows if isinstance(r, dict) for k in ('preset_path', 'sample_path') if r.get(k)]
    if any(dest.resolve() == p.resolve() for p in protected):
        raise ValueError('Choose a new project path, not an input file.')
    header, prefix, base, suffix = _template(fl_dir)
    _replace(prefix, 156, struct.pack('<I', round(bpm * 1000)), required=True)
    _replace(prefix, 194, _text(title), required=True)
    channels = [event for i, row in enumerate(rows) for event in _make_channel(base, row, i)]
    suffix = _configure_mixer(suffix, rows)
    final_header = bytearray(header)
    struct.pack_into('<H', final_header, 2, len(rows))
    events = prefix + channels + suffix
    output = encode_fl(bytes(final_header), events)
    # Atomic replacement means a failed build cannot leave a partial FLP.
    fd, temporary = tempfile.mkstemp(prefix='.flp-builder-', suffix='.tmp', dir=dest.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(output)
        verified_header, verified_events = read_fl(temporary)
        if verified_header != bytes(final_header) or verified_events != events:
            raise ValueError('Generated project failed its structural check.')
        os.replace(temporary, dest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'channels': len(rows), 'warnings': [], 'path': str(dest.resolve())}
