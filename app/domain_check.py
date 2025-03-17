import dns.resolver
import whois
from datetime import datetime
import json
from fuzzywuzzy import fuzz


class DomainChecker:
    def __init__(self, domains_file_path):
        """
        Initializes the DomainChecker with the legitimate domains dataset.

        Parameters:
          domains_file_path (str): Path to the JSON file containing legitimate domains.
        """
        with open(domains_file_path, "r") as file:
            self.legitimate_domains = json.load(file)

    def is_suspicious_domain(self, domain, threshold=80):
        """
        Checks if the domain is suspiciously similar to a legitimate domain.

        Parameters:
          domain (str): The domain to check.
          threshold (int): Similarity threshold (0-100). Domains with a similarity score above this are considered suspicious.

        Returns:
          bool: True if the domain is suspicious, False otherwise.
        """
        for org, domains in self.legitimate_domains.items():
            for legit_domain in domains:
                # Calculate similarity score
                similarity = fuzz.ratio(domain, legit_domain)
                if similarity >= threshold:
                    print(f"Domain {domain} is suspiciously similar to {legit_domain} (similarity: {similarity}%).")
                    return True
        return False

    def get_domain_age(self, domain_info):
        """
        Extracts the domain's age from WHOIS information.

        Parameters:
          domain_info (dict): WHOIS information for the domain.

        Returns:
          int: Domain age in days, or None if the age cannot be determined.
        """
        creation_date = domain_info.creation_date
        updated_date = domain_info.updated_date
        expiration_date = domain_info.expiration_date

        # Handle cases where creation_date is a list
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        # Use creation_date if available
        if creation_date:
            return (datetime.now() - creation_date).days

        # Fallback to updated_date if creation_date is missing
        if updated_date:
            if isinstance(updated_date, list):
                updated_date = updated_date[0]
            return (datetime.now() - updated_date).days

        # Fallback to expiration_date if both creation_date and updated_date are missing
        if expiration_date:
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]
            return (expiration_date - datetime.now()).days

        # If no dates are available, return None
        return None

    def is_legitimate_domain(self, email, min_age_days=365, similarity_threshold=80):
        """
        Checks if the sender's domain is legitimate by:
          1. Checking if the domain is in the list of known legitimate domains.
          2. Checking if the domain is suspiciously similar to a legitimate domain.
          3. Verifying MX records.
          4. Performing a WHOIS lookup.
          5. Checking the domain's age.

        Parameters:
          email (str): The sender's email address.
          min_age_days (int): Minimum age (in days) a domain should be to be considered legitimate.
          similarity_threshold (int): Similarity threshold for detecting suspicious domains.

        Returns:
          bool: True if the domain is considered legitimate, False otherwise.
        """
        # Extract and normalize the domain
        domain = email.split('@')[-1].strip().lower()

        # --- Check if the domain is in the legitimate domains list ---
        for org, domains in self.legitimate_domains.items():
            if domain in domains:
                print(f"Domain {domain} is a known legitimate domain for {org}.")
                return True

        # --- Check for suspicious domains (misspelled or homoglyphs) ---
        if self.is_suspicious_domain(domain, similarity_threshold):
            print(f"Domain {domain} is suspicious.")
            return False

        # --- DNS MX Record Check ---
        try:
            mx_records = dns.resolver.resolve(domain, 'MX')
            if not mx_records:
                print(f"No MX records found for {domain}")
                return False
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.exception.Timeout) as e:
            print(f"MX record check failed for {domain}: {e}")
            return False

        # --- WHOIS Lookup and Domain Age Check ---
        try:
            domain_info = whois.whois(domain)
            domain_age = self.get_domain_age(domain_info)

            if domain_age is None:
                print(f"WHOIS did not return a creation date for {domain}")
                return False

            if domain_age < min_age_days:
                print(f"Domain {domain} is too new: {domain_age} days old.")
                return False

        except Exception as e:
            print(f"WHOIS lookup failed for {domain}: {e}")
            return False

        # If all checks pass, the domain is considered legitimate.
        print(f"Domain {domain} passed legitimacy checks (Age: {domain_age} days).")
        return True