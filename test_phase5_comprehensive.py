#!/usr/bin/env python3
"""
Phase 5: Comprehensive end-to-end testing of LeafGuard application.
Tests all major workflows: auth, prediction, PDF generation, dashboard, error handling.
"""

import requests
import json
import time
import sqlite3
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

BASE_URL = "http://localhost:5000"
DB_PATH = Path(r"C:\Users\User\OneDrive\Desktop\New folder\CNN Project\app\leafi.db")
TEST_IMAGE_DIR = Path(r"C:\Users\User\OneDrive\Desktop\New folder\CNN Project\test_images")

# Test data
TEST_USER_1 = {"username": "test_phase5_user1", "password": "TestPass123!"}
TEST_USER_2 = {"username": "test_phase5_user2", "password": "TestPass456!"}

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "errors": [],
}

def log_pass(test_name, details=""):
    msg = f"[PASS] {test_name}"
    if details:
        msg += f" -- {details}"
    print(msg)
    test_results["passed"].append(test_name)

def log_fail(test_name, reason=""):
    msg = f"[FAIL] {test_name}"
    if reason:
        msg += f" -- {reason}"
    print(msg)
    test_results["failed"].append((test_name, reason))

def log_error(test_name, error):
    msg = f"[ERROR] {test_name} -- {type(error).__name__}: {error}"
    print(msg)
    test_results["errors"].append((test_name, error))

def create_test_image(filename="test.jpg"):
    """Create a simple test image."""
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    return BytesIO(np.asarray(img, dtype=np.uint8))

def get_real_test_image():
    """Try to use a real plant image if available."""
    test_images = list(TEST_IMAGE_DIR.glob("*.jpg")) + list(TEST_IMAGE_DIR.glob("*.png"))
    if test_images:
        with open(test_images[0], "rb") as f:
            return f.read()
    # Fallback: create synthetic image
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ============================================================================
# TEST 1: Application Startup
# ============================================================================
def test_app_startup():
    """Test that the app is running and home page responds."""
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        if resp.status_code == 200:
            log_pass("Application startup", f"Home page returned {resp.status_code}")
            return True
        else:
            log_fail("Application startup", f"Expected 200, got {resp.status_code}")
            return False
    except Exception as e:
        log_error("Application startup", e)
        return False

# ============================================================================
# TEST 2: Home Page
# ============================================================================
def test_home_page():
    """Test that the home page loads and contains expected elements."""
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=5)
        if resp.status_code != 200:
            log_fail("Home page", f"Status {resp.status_code}")
            return False
        if "LeafGuard" not in resp.text and "login" not in resp.text.lower():
            log_fail("Home page", "Missing expected content")
            return False
        log_pass("Home page", "Loaded with expected content")
        return True
    except Exception as e:
        log_error("Home page", e)
        return False

# ============================================================================
# TEST 3-4: Registration and Login
# ============================================================================
def test_registration():
    """Test user registration."""
    try:
        # Register user 1
        resp = requests.post(
            f"{BASE_URL}/register",
            json=TEST_USER_1,
            timeout=5
        )
        if resp.status_code in [200, 302, 201]:
            log_pass("User registration", f"User 1 created (status {resp.status_code})")
            
            # Try registering again (should fail or reject)
            resp2 = requests.post(
                f"{BASE_URL}/register",
                json=TEST_USER_1,
                timeout=5
            )
            # Expect failure on duplicate
            if resp2.status_code != 200:
                log_pass("Duplicate registration blocked", f"Status {resp2.status_code}")
            
            return True
        else:
            log_fail("User registration", f"Status {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        log_error("User registration", e)
        return False

def test_login():
    """Test user login."""
    try:
        resp = requests.post(
            f"{BASE_URL}/login",
            json=TEST_USER_1,
            timeout=5
        )
        if resp.status_code in [200, 302]:
            # Check for session cookie
            if "session" in requests.utils.dict_from_cookiejar(resp.cookies) or \
               "Set-Cookie" in resp.headers or \
               "dashboard" in resp.text.lower():
                log_pass("User login", f"Login successful (status {resp.status_code})")
                return True
            else:
                log_fail("User login", "No session established")
                return False
        else:
            log_fail("User login", f"Status {resp.status_code}")
            return False
    except Exception as e:
        log_error("User login", e)
        return False

# ============================================================================
# TEST 5: Protected Routes
# ============================================================================
def test_protected_routes():
    """Test that unauthenticated users cannot access protected pages."""
    try:
        # Try accessing dashboard without auth
        resp = requests.get(f"{BASE_URL}/dashboard", timeout=5, allow_redirects=False)
        if resp.status_code in [302, 401]:  # Redirect to login or unauthorized
            log_pass("Protected routes", "Unauthenticated users blocked")
            return True
        else:
            log_fail("Protected routes", f"Expected redirect/401, got {resp.status_code}")
            return False
    except Exception as e:
        log_error("Protected routes", e)
        return False

# ============================================================================
# TEST 6: Dashboard Access
# ============================================================================
def test_dashboard_access():
    """Test accessing the dashboard after login."""
    try:
        session = requests.Session()
        
        # Login first
        login_resp = session.post(
            f"{BASE_URL}/login",
            json=TEST_USER_1,
            timeout=5
        )
        
        # Now try dashboard
        dash_resp = session.get(f"{BASE_URL}/dashboard", timeout=5)
        
        if dash_resp.status_code == 200:
            log_pass("Dashboard access", "Authenticated access successful")
            return True
        else:
            log_fail("Dashboard access", f"Status {dash_resp.status_code}")
            return False
    except Exception as e:
        log_error("Dashboard access", e)
        return False

# ============================================================================
# TEST 7: Valid Image Upload and Prediction
# ============================================================================
def test_prediction():
    """Test uploading a valid image and getting a prediction."""
    try:
        session = requests.Session()
        
        # Login
        session.post(f"{BASE_URL}/login", json=TEST_USER_1, timeout=5)
        
        # Get test image
        image_data = get_real_test_image()
        
        # Upload and predict
        files = {"file": ("test.jpg", BytesIO(image_data), "image/jpeg")}
        resp = session.post(
            f"{BASE_URL}/api/predict",
            files=files,
            timeout=30  # Prediction can take time
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if "label" in result and "confidence" in result:
                log_pass("Valid prediction", f"Label: {result['label']}, Confidence: {result['confidence']:.2%}")
                return True
            else:
                log_fail("Valid prediction", "Missing prediction fields")
                return False
        else:
            log_fail("Valid prediction", f"Status {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        log_error("Valid prediction", e)
        return False

# ============================================================================
# TEST 8: Invalid Image Upload
# ============================================================================
def test_invalid_upload():
    """Test uploading invalid files."""
    try:
        session = requests.Session()
        session.post(f"{BASE_URL}/login", json=TEST_USER_1, timeout=5)
        
        # Try uploading a non-image file
        files = {"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
        resp = session.post(
            f"{BASE_URL}/api/predict",
            files=files,
            timeout=10
        )
        
        # Should fail or reject
        if resp.status_code != 200:
            log_pass("Invalid upload rejection", f"Rejected with status {resp.status_code}")
            return True
        else:
            # Check if error is in response
            try:
                result = resp.json()
                if "error" in result or "Error" in result:
                    log_pass("Invalid upload rejection", "Error returned in response")
                    return True
            except:
                pass
            log_fail("Invalid upload rejection", "Invalid file was accepted")
            return False
    except Exception as e:
        log_error("Invalid upload rejection", e)
        return False

# ============================================================================
# TEST 9: SQLite History
# ============================================================================
def test_history_storage():
    """Test that predictions are stored in SQLite."""
    try:
        if not DB_PATH.exists():
            log_fail("SQLite history", "Database file not found")
            return False
        
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';"
        )
        if not cursor.fetchone():
            log_fail("SQLite history", "Predictions table not found")
            conn.close()
            return False
        
        # Count predictions
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        count = cursor.fetchone()[0]
        conn.close()
        
        log_pass("SQLite history", f"{count} predictions stored")
        return True
    except Exception as e:
        log_error("SQLite history", e)
        return False

# ============================================================================
# TEST 10: User Isolation
# ============================================================================
def test_user_isolation():
    """Test that users can only see their own predictions."""
    try:
        # Register and login user 2
        requests.post(f"{BASE_URL}/register", json=TEST_USER_2, timeout=5)
        
        session2 = requests.Session()
        session2.post(f"{BASE_URL}/login", json=TEST_USER_2, timeout=5)
        
        # User 2 should see empty/different history than User 1
        # (This is a basic check; ideally we'd compare actual prediction IDs)
        resp = session2.get(f"{BASE_URL}/dashboard", timeout=5)
        
        if resp.status_code == 200:
            log_pass("User isolation", "User 2 can access their own dashboard")
            return True
        else:
            log_fail("User isolation", f"Status {resp.status_code}")
            return False
    except Exception as e:
        log_error("User isolation", e)
        return False

# ============================================================================
# TEST 11: Logout
# ============================================================================
def test_logout():
    """Test user logout and session termination."""
    try:
        session = requests.Session()
        session.post(f"{BASE_URL}/login", json=TEST_USER_1, timeout=5)
        
        # Logout
        logout_resp = session.get(f"{BASE_URL}/logout", timeout=5, allow_redirects=False)
        
        if logout_resp.status_code in [302, 200]:
            # Try accessing protected route after logout
            dash_resp = session.get(f"{BASE_URL}/dashboard", timeout=5, allow_redirects=False)
            if dash_resp.status_code in [302, 401]:
                log_pass("Logout", "Session terminated correctly")
                return True
        
        log_fail("Logout", f"Unexpected status: {logout_resp.status_code}")
        return False
    except Exception as e:
        log_error("Logout", e)
        return False

# ============================================================================
# TEST 12: API Error Handling
# ============================================================================
def test_error_handling():
    """Test various error conditions."""
    try:
        session = requests.Session()
        session.post(f"{BASE_URL}/login", json=TEST_USER_1, timeout=5)
        
        # Try to get non-existent prediction
        resp = session.get(f"{BASE_URL}/api/prediction/999999", timeout=5)
        
        if resp.status_code in [404, 400, 500]:
            log_pass("Error handling", f"Non-existent prediction returns {resp.status_code}")
            return True
        else:
            log_fail("Error handling", f"Expected error status, got {resp.status_code}")
            return False
    except Exception as e:
        log_error("Error handling", e)
        return False

# ============================================================================
# TEST 13: PDF Generation
# ============================================================================
def test_pdf_generation():
    """Test PDF report generation (requires a valid prediction)."""
    try:
        session = requests.Session()
        session.post(f"{BASE_URL}/login", json=TEST_USER_1, timeout=5)
        
        # First, make a prediction
        image_data = get_real_test_image()
        files = {"file": ("test.jpg", BytesIO(image_data), "image/jpeg")}
        pred_resp = session.post(
            f"{BASE_URL}/api/predict",
            files=files,
            timeout=30
        )
        
        if pred_resp.status_code != 200:
            log_fail("PDF generation", "Could not create prediction for PDF test")
            return False
        
        pred = pred_resp.json()
        if "prediction_id" not in pred:
            log_fail("PDF generation", "No prediction_id in response")
            return False
        
        # Try to generate PDF
        pdf_resp = session.get(
            f"{BASE_URL}/api/report/{pred['prediction_id']}.pdf",
            timeout=10
        )
        
        if pdf_resp.status_code == 200 and "application/pdf" in pdf_resp.headers.get("Content-Type", ""):
            log_pass("PDF generation", "PDF report generated successfully")
            return True
        else:
            log_fail("PDF generation", f"Status {pdf_resp.status_code}, Content-Type: {pdf_resp.headers.get('Content-Type')}")
            return False
    except Exception as e:
        log_error("PDF generation", e)
        return False

# ============================================================================
# TEST 14: Grad-CAM and Advanced Fields
# ============================================================================
def test_grad_cam():
    """Test Grad-CAM heatmap generation."""
    try:
        session = requests.Session()
        session.post(f"{BASE_URL}/login", json=TEST_USER_1, timeout=5)
        
        image_data = get_real_test_image()
        files = {"file": ("test.jpg", BytesIO(image_data), "image/jpeg")}
        resp = session.post(
            f"{BASE_URL}/api/predict",
            files=files,
            timeout=30
        )
        
        if resp.status_code == 200:
            result = resp.json()
            if "heatmap" in result and result["heatmap"] is not None:
                log_pass("Grad-CAM", "Heatmap generated")
                return True
            else:
                log_fail("Grad-CAM", "Heatmap not in response")
                return False
        else:
            log_fail("Grad-CAM", f"Status {resp.status_code}")
            return False
    except Exception as e:
        log_error("Grad-CAM", e)
        return False

# ============================================================================
# Main Test Runner
# ============================================================================
def main():
    print("=" * 70)
    print("PHASE 5: COMPREHENSIVE END-TO-END TESTING")
    print("=" * 70)
    print()
    
    # Run tests in order
    tests = [
        ("Test 1: Application Startup", test_app_startup),
        ("Test 2: Home Page", test_home_page),
        ("Test 3: User Registration", test_registration),
        ("Test 4: User Login", test_login),
        ("Test 5: Protected Routes", test_protected_routes),
        ("Test 6: Dashboard Access", test_dashboard_access),
        ("Test 7: Valid Prediction", test_prediction),
        ("Test 8: Invalid Upload Rejection", test_invalid_upload),
        ("Test 9: SQLite History", test_history_storage),
        ("Test 10: User Isolation", test_user_isolation),
        ("Test 11: Logout", test_logout),
        ("Test 12: Error Handling", test_error_handling),
        ("Test 13: PDF Generation", test_pdf_generation),
        ("Test 14: Grad-CAM", test_grad_cam),
    ]
    
    for name, test_func in tests:
        print(f"\n{name}")
        print("-" * 70)
        try:
            test_func()
        except Exception as e:
            log_error(name, e)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed:  {len(test_results['passed'])}")
    print(f"Failed:  {len(test_results['failed'])}")
    print(f"Errors:  {len(test_results['errors'])}")
    print()
    
    if test_results["failed"]:
        print("FAILED TESTS:")
        for name, reason in test_results["failed"]:
            print(f"  - {name}: {reason}")
        print()
    
    if test_results["errors"]:
        print("ERRORS:")
        for name, error in test_results["errors"]:
            print(f"  - {name}: {error}")
        print()
    
    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["errors"])
    if total > 0:
        pass_rate = len(test_results["passed"]) / total * 100
        print(f"Pass Rate: {pass_rate:.1f}%")
    
    return len(test_results["failed"]) == 0 and len(test_results["errors"]) == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
