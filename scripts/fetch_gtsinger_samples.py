#!/usr/bin/env python3
"""Fetch an explicit revision-pinned P1 sample selection, without overwrites."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


REPOSITORY = "AaronZ345/GTSinger"
REVISION = "d78cd68d00135b1995da03f25fccb3a221efd8c1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data/gtsinger-samples.json"
STEMS = (
    "Chinese/ZH-Alto-1/Breathy/不再见/Control_Group/0000",
    "English/EN-Alto-1/Glissando/my love/Control_Group/0000",
)
REPRESENTATIVE_BASES = (
    ("Chinese/ZH-Alto-1/Breathy/不再见", "Breathy_Group"),
    ("Chinese/ZH-Alto-1/Breathy/慢慢喜欢你", "Breathy_Group"),
    ("Chinese/ZH-Tenor-1/Breathy/慢慢喜欢你", "Breathy_Group"),
    ("Chinese/ZH-Tenor-1/Glissando/给电影人的情书", "Glissando_Group"),
    ("English/EN-Alto-1/Glissando/my love", "Glissando_Group"),
    ("English/EN-Alto-2/Glissando/My Love", "Glissando_Group"),
    ("English/EN-Alto-1/Glissando/rolling in the deep", "Glissando_Group"),
    ("English/EN-Tenor-1/Glissando/Rolling in the Deep", "Glissando_Group"),
    ("English/EN-Alto-2/Glissando/Someone Like You", "Glissando_Group"),
    ("English/EN-Tenor-1/Glissando/Someone Like You", "Glissando_Group"),
)
REPRESENTATIVE_STEMS = tuple(base + "/" + group + "/0000"
                             for base, technique_group in REPRESENTATIVE_BASES
                             for group in ("Control_Group", technique_group))


def selection_paths(stems):
    paths = {"dataset_license.md"}
    for stem in stems:
        paths.update(stem + ext for ext in (".wav", ".json", ".musicxml", ".TextGrid", "_TextGrid"))
    return paths


ALLOWED_PATHS = selection_paths(STEMS + REPRESENTATIVE_STEMS)
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
REPRESENTATIVE_MAX_TOTAL_BYTES = 100_000_000
# Exact CDN origin observed in Hugging Face's resolver response for these WAVs.
ALLOWED_HOSTS = {"huggingface.co", "us.aws.cdn.hf.co"}


def safe_destination(root, relative_path):
    path = PurePosixPath(relative_path)
    if (not relative_path or path.is_absolute() or "\\" in relative_path
            or any(part in ("", ".", "..") for part in relative_path.split("/"))):
        raise ValueError("Unsafe relative path: " + repr(relative_path))
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("Input root must not be a symlink")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("Refusing symlink in destination: " + str(target))
    return target


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object")
    if manifest.get("repository") != REPOSITORY or manifest.get("revision") != REVISION:
        raise ValueError("Only the pinned GTSinger repository/revision is supported")
    selection = manifest.get("selection", "smoke")
    if selection == "smoke":
        stems, byte_limit = STEMS, MAX_TOTAL_BYTES
    elif selection == "representative-v1":
        stems, byte_limit = REPRESENTATIVE_STEMS, REPRESENTATIVE_MAX_TOTAL_BYTES
    else:
        raise ValueError("Only the fixed smoke or representative-v1 selection is supported")
    files = manifest.get("files", [])
    if not isinstance(files, list) or not all(isinstance(entry, dict) for entry in files):
        raise ValueError("Manifest files must be objects")
    paths = [entry.get("path") for entry in files]
    if not all(isinstance(value, str) for value in paths) or len(set(paths)) != len(paths):
        raise ValueError("Manifest paths must be unique strings")
    required = {"dataset_license.md"}
    for stem in stems:
        required.update(stem + suffix for suffix in (".wav", ".json", ".musicxml"))
        textgrids = {stem + ".TextGrid", stem + "_TextGrid"} & set(paths)
        if len(textgrids) != 1:
            raise ValueError("Each approved sample must have exactly one TextGrid")
        required.update(textgrids)
    if set(paths) != required:
        raise ValueError("Manifest must contain exactly the approved sample quartets and license")
    samples = manifest.get("samples", [])
    if (not isinstance(samples, list) or not all(isinstance(s, dict) for s in samples)
            or len(samples) != len(stems)
            or {s.get("relative_stem") for s in samples if isinstance(s.get("relative_stem"), str)} != set(stems)):
        raise ValueError("Manifest must contain exactly the fixed approved samples")
    for sample in samples:
        if not isinstance(sample.get("song_id"), str) or not sample["song_id"].strip():
            raise ValueError("Each sample needs a candidate song_id")
        if sample.get("split", "unassigned") != "unassigned":
            raise ValueError("Sample acquisition cannot assign dataset splits")
        if selection == "representative-v1" and sample.get("song_identity_review_status") != "pending":
            raise ValueError("Representative song identities are unreviewed candidates")
    if manifest.get("license", {}).get("path") != "dataset_license.md":
        raise ValueError("The source license must be retained")
    for entry in files:
        size = entry.get("size")
        if type(size) is not int or not 0 < size <= MAX_FILE_BYTES:
            raise ValueError("Invalid or excessive file size")
        hash_key = "sha256" if "sha256" in entry else "git_blob_sha1"
        if not re.fullmatch(r"[0-9a-f]{%d}" % (64 if hash_key == "sha256" else 40),
                            entry.get(hash_key, "")):
            raise ValueError("Missing or invalid source checksum")
    if sum(entry["size"] for entry in files) > byte_limit:
        raise ValueError("Manifest exceeds total byte limit")
    return manifest


def source_url(entry):
    if entry["path"] not in ALLOWED_PATHS:
        raise ValueError("File is outside the approved sample selection")
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (
        REPOSITORY, REVISION, quote(entry["path"], safe="/"))


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS
            or parsed.port not in (None, 443) or parsed.username or parsed.password
            or parsed.fragment):
        raise ValueError("Refusing non-approved download origin")


class RestrictedRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def verify_file(path, entry):
    """Verify HF LFS SHA-256 or Git blob SHA-1, and retain a content SHA-256."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size != entry["size"]:
        raise ValueError("Source size/type mismatch: " + entry["path"])
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(("blob %d\0" % entry["size"]).encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            sha256.update(chunk)
            git_sha1.update(chunk)
    actual = sha256.hexdigest() if "sha256" in entry else git_sha1.hexdigest()
    expected = entry.get("sha256", entry.get("git_blob_sha1"))
    if actual != expected:
        raise ValueError("Source checksum mismatch: " + entry["path"])
    return sha256.hexdigest()


def download_file(root, entry, opener):
    target = safe_destination(root, entry["path"])
    url = source_url(entry)
    if target.exists():
        return {"path": entry["path"], "sha256": verify_file(target, entry),
                "source_url": url, "status": "reused"}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False) as output:
            temporary = Path(output.name)
            request = Request(url, headers={"User-Agent": "auto-lrc-p1-sample-fetch/1", "Accept-Encoding": "identity"})
            with opener.open(request, timeout=30) as response:
                validate_url(response.geturl())
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != entry["size"]:
                    raise ValueError("Remote size differs from the pinned manifest")
                count = 0
                while True:
                    chunk = response.read(min(64 * 1024, entry["size"] + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > entry["size"] or count > MAX_FILE_BYTES:
                        raise ValueError("Download exceeded the declared byte limit")
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        digest = verify_file(temporary, entry)
        safe_destination(root, entry["path"])
        # Atomic no-clobber publication: rename/replace would overwrite a concurrent file.
        os.link(temporary, target)
        return {"path": entry["path"], "sha256": digest, "source_url": url, "status": "downloaded"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_samples(input_root, manifest_path=DEFAULT_MANIFEST, *, opener=None):
    manifest = load_manifest(manifest_path)
    root = Path(input_root).absolute()
    if root.is_symlink():
        raise ValueError("Input root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    opener = opener or build_opener(RestrictedRedirects())
    records = [download_file(root, entry, opener) for entry in manifest["files"]]
    provenance = {"repository": REPOSITORY, "revision": REVISION,
                  "selection": manifest.get("selection", "smoke"),
                  "license": manifest["license"], "files": records,
                  "total_bytes": sum(entry["size"] for entry in manifest["files"])}
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(fetch_samples(args.input_root, args.manifest), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, "GTSinger download failed: %s\n" % error)


if __name__ == "__main__":
    main()
